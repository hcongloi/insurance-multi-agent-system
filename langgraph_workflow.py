# langgraph_workflow.py
import operator
import re
import json
from typing import TypedDict, Annotated, List, Union, Dict, Any
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME

from agents.customer_agent import create_customer_agent
from agents.lead_agent import create_lead_agent
from agents.knowledge_agent import create_knowledge_agent
from tools.crm_tool import get_customer_info 
from tools.recommendation_tool import generate_insurance_recommendations
from utils.rag_pipeline import ingest_and_get_vector_store, get_persisted_vector_store, CHROMA_DB_DIR


# --- RAG INITIALIZATION ---
_embeddings_instance = None
_vector_store_instance = None

def get_global_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set. Cannot initialize GoogleGenerativeAIEmbeddings.")
        _embeddings_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return _embeddings_instance

def get_global_vector_store() -> Chroma: 
    global _vector_store_instance
    if _vector_store_instance is None:
        try:
            _vector_store_instance = get_persisted_vector_store(get_global_embeddings())
            print(f"Chroma DB loaded from '{CHROMA_DB_DIR}'.")
        except FileNotFoundError:
            print(f"Chroma DB not found at '{CHROMA_DB_DIR}'. Ingesting documents for the first time...")
            _vector_store_instance = ingest_and_get_vector_store(get_global_embeddings())
    return _vector_store_instance

# Perform initial ingestion/loading of RAG components when module is imported
try:
    get_global_vector_store()
except Exception as e:
    print(f"ERROR: Failed to initialize Chroma DB at startup: {e}. RAG might not work.")
# --- END RAG INITIALIZATION ---


# 1. Define AgentState
class AgentState(TypedDict):
    """
    Represents the state of our graph.
    """
    input: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    # agent_outcome is for internal ReAct trace, not part of our custom state passing
    # intermediate_steps is also part of ReAct trace, will be extracted safely
    
    customer_info_result: str 
    lead_info_result: str     
    kb_info_result: str       
    
    customer_profile: Dict[str, Any] 
    available_products_kb: str      
    recommendation_result: str      
    
    is_recommendation_flow: bool 

    final_response: str
    error_message: str 
    
    # Store the router's decision explicitly for conditional edges
    router_decision: str

# 2. Create Agent Executors
customer_agent_executor = create_customer_agent()
lead_agent_executor = create_lead_agent()
knowledge_agent_executor = create_knowledge_agent(get_global_embeddings(), get_global_vector_store())


# 3. Define Nodes for the Graph
def run_customer_agent_node(state: AgentState):
    print("---EXECUTING CUSTOMER AGENT---")
    try:
        # AgentExecutor's invoke returns a dict with 'output' and optionally 'intermediate_steps'
        result = customer_agent_executor.invoke({"input": state["input"]})
        
        customer_info_output = result.get("output", "")
        agent_intermediate_steps = result.get("intermediate_steps", [])

        customer_profile_data = {}
        if state.get("is_recommendation_flow", False) and \
            customer_info_output and "customer not found" not in customer_info_output.lower() \
            and "could not be found" not in customer_info_output.lower() \
            and "i cannot find any customer" not in customer_info_output.lower():
            
            customer_identifier = ""
            
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", state["input"])
            if email_match:
                customer_identifier = email_match.group(0)
            else:
                id_match = re.search(r"(cust\d{3})", state["input"].lower())
                if id_match:
                    customer_identifier = id_match.group(0).upper()
                else: 
                    name_extractor_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
                    name_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Extract the full name of the customer from the query. If no specific full name is clearly mentioned, respond with 'NONE'. Example: 'Find customer John Doe' -> 'John Doe'. 'Customer with email' -> 'NONE'"),
                        ("human", "{query}")
                    ])
                    name_chain = name_prompt | name_extractor_llm
                    extracted_name = name_chain.invoke({"query": state["input"]}).content.strip()
                    if extracted_name.lower() != "none" and len(extracted_name.split()) >= 2:
                        customer_identifier = extracted_name
            
            if customer_identifier:
                customer_profile_data = get_customer_info.invoke(customer_identifier) 
                print(f"---Extracted customer profile for recommendation: {customer_profile_data.get('name')}---")

        return {
            "customer_info_result": customer_info_output, 
            "intermediate_steps": agent_intermediate_steps, # Access safely
            "customer_profile": customer_profile_data, 
            "is_recommendation_flow": state.get("is_recommendation_flow", False),
            "router_decision": state.get("router_decision") # Pass router decision along
        }
    except Exception as e:
        error_msg = f"Error in customer agent: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "customer_info_result": "",
            "error_message": error_msg,
            "customer_profile": {},
            "is_recommendation_flow": state.get("is_recommendation_flow", False),
            "router_decision": state.get("router_decision")
        }

def run_lead_agent_node(state: AgentState):
    print("---EXECUTING LEAD AGENT---")
    try:
        result = lead_agent_executor.invoke({"input": state["input"]})
        return {
            "lead_info_result": result.get("output", ""), # Access safely
            "intermediate_steps": result.get("intermediate_steps", []), # Access safely
            "is_recommendation_flow": state.get("is_recommendation_flow", False),
            "router_decision": state.get("router_decision")
        }
    except Exception as e:
        error_msg = f"Error in lead agent: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "lead_info_result": "",
            "error_message": error_msg,
            "is_recommendation_flow": state.get("is_recommendation_flow", False),
            "router_decision": state.get("router_decision")
        }

def run_knowledge_agent_node(state: AgentState):
    print("---EXECUTING KNOWLEDGE AGENT---")
    try:
        kb_input = state["input"]
        if state.get("is_recommendation_flow", False):
            kb_input = "Tell me about all insurance products" 
        
        result = knowledge_agent_executor.invoke({"input": kb_input})

        return {
            "kb_info_result": result.get("output", ""), # Access safely
            "intermediate_steps": result.get("intermediate_steps", []), # Access safely
            "available_products_kb": result.get("output", ""), # Access safely
            "is_recommendation_flow": state.get("is_recommendation_flow", False),
            "router_decision": state.get("router_decision")
        }
    except Exception as e:
        error_msg = f"Error in knowledge agent: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "kb_info_result": "",
            "error_message": error_msg,
            "available_products_kb": "",
            "is_recommendation_flow": state.get("is_recommendation_flow", False),
            "router_decision": state.get("router_decision")
        }

def run_recommendation_node(state: AgentState):
    print("---GENERATING RECOMMENDATIONS---")
    try:
        customer_profile_dict = state.get("customer_profile", {})
        if not isinstance(customer_profile_dict, dict) or not customer_profile_dict:
            return {"recommendation_result": "No valid customer profile available for recommendation."}

        customer_profile_json = json.dumps(customer_profile_dict)
        
        recommendation_output = generate_insurance_recommendations.invoke({
            "customer_profile_json": customer_profile_json,
            "available_products_kb": state.get("available_products_kb", "")
        })
        return {"recommendation_result": recommendation_output}
    except Exception as e:
        error_msg = f"Error in recommendation generation: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "recommendation_result": "",
            "error_message": error_msg
        }

def generate_final_response_node(state: AgentState):
    print("---GENERATING FINAL RESPONSE (ORCHESTRATOR'S AGGREGATION)---")
    response_parts = []
    
    error_msg = state.get("error_message", "").strip()
    if error_msg:
        response_parts.append(f"⚠️ **An internal error occurred:** {error_msg}")
    
    # Kiểm tra xem có phải recommendation flow không
    is_rec_flow = state.get("is_recommendation_flow", False)
    recommendation_res = state.get("recommendation_result", "").strip()
    
    # Nếu là recommendation flow và có kết quả recommendation
    if is_rec_flow and recommendation_res:
        # Kiểm tra xem recommendation có hợp lệ không
        invalid_phrases = [
            "no valid customer profile available",
            "no customer profile provided",
            "no product knowledge base information provided",
            "invalid customer profile json format",
            "error in recommendation"
        ]
        
        is_valid_recommendation = not any(phrase in recommendation_res.lower() for phrase in invalid_phrases)
        
        if is_valid_recommendation:
            # Hiển thị recommendation trước
            response_parts.append(f"## 🌟 Insurance Recommendations\n\n{recommendation_res}\n")
            
            # Sau đó hiển thị customer details (nhưng loại bỏ phần "unable to recommend")
            customer_res = state.get("customer_info_result", "").strip()
            if customer_res:
                # Loại bỏ các dòng về "unable to recommend"
                customer_lines = customer_res.split('\n')
                filtered_lines = [
                    line for line in customer_lines 
                    if not any(phrase in line.lower() for phrase in [
                        "unable to recommend",
                        "cannot recommend",
                        "i am unable to",
                        "i cannot recommend"
                    ])
                ]
                filtered_customer_info = '\n'.join(filtered_lines).strip()
                
                if filtered_customer_info:
                    response_parts.append(f"## 👤 Customer Profile\n\n{filtered_customer_info}\n")
            
            return {"final_response": "\n".join(response_parts)}
    
    # Nếu không phải recommendation flow hoặc recommendation failed
    customer_res = state.get("customer_info_result", "").strip()
    if customer_res and not any(phrase in customer_res.lower() for phrase in [
        "customer not found",
        "could not be found",
        "i cannot find any customer"
    ]):
        response_parts.append(f"## 👤 Customer Information\n\n{customer_res}")
    
    lead_res = state.get("lead_info_result", "").strip()
    if lead_res and not any(phrase in lead_res.lower() for phrase in [
        "no leads matching the criteria were found",
        "no leads found"
    ]):
        response_parts.append(f"## 📊 Lead Information\n\n{lead_res}")

    kb_res = state.get("kb_info_result", "").strip()
    # Chỉ hiển thị KB result nếu KHÔNG phải recommendation flow
    if not is_rec_flow and kb_res and not any(phrase in kb_res.lower() for phrase in [
        "no specific information found",
        "no knowledge base content available",
        "i cannot find the answer"
    ]):
        response_parts.append(f"## 📚 Knowledge Base Info\n\n{kb_res}")
    
    if not response_parts and not error_msg:
        final_msg = "I couldn't find relevant information for your query using the available tools. Please rephrase or provide more details."
    else:
        final_msg = "\n\n".join(response_parts)
        
    if not final_msg and error_msg:
        final_msg = f"⚠️ **An internal error occurred:** {error_msg}"
    
    return {"final_response": final_msg}

# Helper function to determine the routing target based on LLM's decision
def _determine_routing_target(state: AgentState) -> str:
    """Uses LLM to classify intent and returns the target node name string."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert routing assistant (Orchestrator). Your task is to analyze the user's query and determine
        the primary intent to route it to the most suitable specialized agent or workflow.
        Reply with ONLY ONE of the following keywords: "customer", "lead", "knowledge", "recommendation_workflow", or "general".
        
        - Use "customer" for queries directly about existing customers, their policies, or history (e.g., "Find customer John Doe", "What are CUST001's policies?", "Email of Jane Doe").
        - Use "lead" for queries about potential leads, sales prospects, lead scores, or lead lists (e.g., "Find qualified leads", "Leads interested in auto insurance", "Show me leads in California").
        - Use "knowledge" for general questions about insurance products, definitions, policy types, or FAQs (e.g., "What is life insurance?", "Explain comprehensive coverage", "What is a premium?").
        - Use "recommendation_workflow" if the query explicitly asks to find customer info AND recommend products based on that profile (e.g., "Find customer John Doe and recommend insurance products based on his profile", "Recommend coverage for Sarah Johnson").
        - Use "general" if the query doesn't fit any of the above categories or is a general conversational question.
        """),
        ("human", "{input}")
    ])
    
    router_chain = router_prompt | llm
    
    try:
        response = router_chain.invoke({"input": state["input"]}).content.lower().strip()
    except Exception as e:
        print(f"ERROR in LLM router: {e}. Defaulting to knowledge agent.")
        response = "general"
        
    print(f"---ORCHESTRATOR DECISION: {response}---")

    if "recommendation_workflow" in response:
        return "set_recommendation_flag"
    elif "customer" in response:
        return "customer_agent_node"
    elif "lead" in response:
        return "lead_agent_node"
    elif "knowledge" in response:
        return "knowledge_agent_node"
    else:
        return "knowledge_agent_node"

# This is the actual NODE function that will update AgentState
def run_router_node(state: AgentState):
    print("---ORCHESTRATOR: INTENT CLASSIFICATION & ROUTING NODE---")
    # Call the helper function to get the target node name
    target_node_name = _determine_routing_target(state)
    return {"router_decision": target_node_name}


def set_recommendation_flag_node(state: AgentState):
    print("---ORCHESTRATOR: SETTING RECOMMENDATION FLAG---")
    # Preserve original router_decision from previous node
    return {"is_recommendation_flow": True, "router_decision": state.get("router_decision")}


# 5. Build the Graph
def create_multi_agent_workflow():
    workflow = StateGraph(AgentState)

    # Add ALL nodes (Orchestrator manages these specialized agents)
    workflow.add_node("customer_agent_node", run_customer_agent_node)
    workflow.add_node("lead_agent_node", run_lead_agent_node)
    workflow.add_node("knowledge_agent_node", run_knowledge_agent_node)
    workflow.add_node("run_recommendation_node", run_recommendation_node) 
    workflow.add_node("final_response_node", generate_final_response_node)
    
    # The new router node
    workflow.add_node("router_node", run_router_node)


    # Specific nodes for recommendation flow setup
    workflow.add_node("set_recommendation_flag", set_recommendation_flag_node)
    
    def prepare_kb_query_for_recommendation_node(state: AgentState):
        print("---ORCHESTRATOR: PREPARING KB QUERY FOR RECOMMENDATION---")
        # Overwrite input with a general KB query for product info
        # Keep chat_history if needed for context in future steps, but not used here.
        return {
            "input": "Tell me about all insurance products", 
            "chat_history": state["chat_history"],
            "router_decision": state.get("router_decision") # Pass router decision
        }
    workflow.add_node("prepare_kb_query_for_recommendation", prepare_kb_query_for_recommendation_node)


    # Set the general entry point (Orchestrator starts here)
    workflow.set_entry_point("router_node") # Router is now the entry point

    # Intent Classification & Initial Routing
    # This conditional edge reads the router_decision from the state
    workflow.add_conditional_edges(
        "router_node",
        lambda state: state["router_decision"], # Read decision from state
        {
            "customer_agent_node": "customer_agent_node", 
            "lead_agent_node": "lead_agent_node",
            "knowledge_agent_node": "knowledge_agent_node", 
            "set_recommendation_flag": "set_recommendation_flag", 
        },
    )

    # Workflow Coordination: Recommendation Flow
    workflow.add_edge("set_recommendation_flag", "customer_agent_node") # Execute CustomerAgent for profile
    
    # After CustomerAgent: If it's a recommendation flow AND customer profile is found, get KB. Else, generate final response.
    workflow.add_conditional_edges(
        "customer_agent_node",
        lambda state: "prepare_kb_query_for_recommendation" 
            if state.get("is_recommendation_flow", False) and state.get("customer_profile") and state["customer_profile"] 
            else "final_response_node",
        {
            "prepare_kb_query_for_recommendation": "prepare_kb_query_for_recommendation",
            "final_response_node": "final_response_node"
        }
    )
    
    workflow.add_edge("prepare_kb_query_for_recommendation", "knowledge_agent_node") # Get general product info

    # After KnowledgeAgent: If it's a recommendation flow AND all data is ready, run recommendation. Else, generate final response.
    workflow.add_conditional_edges(
        "knowledge_agent_node",
        lambda state: "run_recommendation_node" 
            if state.get("is_recommendation_flow", False) and state.get("available_products_kb") and state["available_products_kb"] and state.get("customer_profile") and state["customer_profile"]
            else "final_response_node",
        {
            "run_recommendation_node": "run_recommendation_node",
            "final_response_node": "final_response_node" 
        }
    )

    workflow.add_edge("run_recommendation_node", "final_response_node") # Generate final recommendation

    # Workflow Coordination: Single Agent Paths (direct to final response)
    workflow.add_edge("lead_agent_node", "final_response_node")
    
    # Response Aggregation and Delivery
    workflow.add_edge("final_response_node", END)

    return workflow.compile()

if __name__ == "__main__":
    app = create_multi_agent_workflow()

    print("\n--- Testing Multi-Agent Workflow ---")

    test_queries = [
        "Find customer with email john@example.com",
        "Show me qualified leads in Texas",
        "What is comprehensive auto insurance?",
        "Who is Alice Williams?",
        "Are there any new leads interested in life insurance?",
        "Explain different types of life insurance.",
        "Tell me about CUST003's policies.",
        "What is a premium?",
        "Find leads with score above 80 interested in auto insurance.",
        "Find customer John Doe and recommend insurance products based on his profile",
        "Find customer Emily Brown and recommend insurance products based on her profile",
        "Recommend products for non_existent@example.com",
        "Can you tell me a joke?",
        "What is an insurance deductible?",
        "Show me customer John Doe's current policies and recommend additional coverage options",
        "Recommend coverage for John Doe",
    ]

    for query in test_queries:
        print(f"\n--- USER QUERY: {query} ---")
        initial_state = {
            "input": query, 
            "chat_history": [], 
            "customer_info_result": "", 
            "lead_info_result": "", 
            "kb_info_result": "", 
            "customer_profile": {}, 
            "available_products_kb": "", 
            "recommendation_result": "",
            "final_response": "", 
            "intermediate_steps": [],
            "is_recommendation_flow": False,
            "error_message": "",
            "router_decision": "" # Initialize router_decision
        }
        for s in app.stream(initial_state, stream_mode="updates"):
            if "__end__" not in s:
                for key, value in s.items():
                    print(f"  Changed: {key}")
            else:
                print(f"Final Response: {s['__end__'].get('final_response','')}")
        print("\n" + "="*80 + "\n")