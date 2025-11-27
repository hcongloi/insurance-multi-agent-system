# main.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time  # Added to measure response time and support small loading animations

# Import create_multi_agent_workflow from langgraph_workflow.py
from langgraph_workflow import create_multi_agent_workflow

# Use st.cache_resource so the LangGraph app is initialized only once.
# This avoids re-initializing models and vector stores on every Streamlit rerun.
@st.cache_resource
def get_langgraph_app():
    return create_multi_agent_workflow()

app = get_langgraph_app()

# Configure the Streamlit page
st.set_page_config(page_title="Multi-Agent Insurance Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Insurance Assistant")
st.caption("Powered by LangChain 1.0.5 & Google Gemini 1.5 Flash")

# Initialize session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_execution_log" not in st.session_state:
    st.session_state.agent_execution_log = []

def get_response(user_query: str) -> str:
    """
    Execute the multi-agent workflow and stream results.
    Log intermediate steps to st.session_state.agent_execution_log.
    Return the final aggregated response as a string.
    """
    st.session_state.agent_execution_log = []  # Reset the log for each new query
    
    # Create the initial state for the workflow. It's important to use a new dict each time.
    inputs = {
        "input": user_query,
        "chat_history": [],  # Pass an empty chat_history (AgentState uses Annotated[List[BaseMessage], operator.add])
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
        "router_decision": ""  # Initialize router_decision
    }
    
    full_response = ""
    last_state = None  # Variable to store the last state seen
    start_time = time.time()  # Start response time measurement
    
    try:
        # Stream execution updates so we can monitor node-level progress
        for s in app.stream(inputs, stream_mode="updates"):
            # stream_mode="updates" yields a dict with node name keys and state update values
            for key, value in s.items():
                # Keep the last state from each update
                if isinstance(value, dict):
                    if last_state is None:
                        last_state = value.copy()
                    else:
                        last_state.update(value)
                
                # Log for the router node
                if key == "router_node":
                    router_decision = value.get("router_decision") 
                    if router_decision:
                        st.session_state.agent_execution_log.append(f"🔄 **Orchestrator Routing:** Decided to use `{router_decision}`")
                
                # Log when the recommendation flag node runs
                elif key == "set_recommendation_flag":
                    st.session_state.agent_execution_log.append(f"🚩 **Orchestrator Flag:** `is_recommendation_flow` set to `True`.")
                
                # Log for agent nodes and other nodes
                elif key.endswith("_agent_node") or key == "run_recommendation_node" or key == "prepare_kb_query_for_recommendation":
                    agent_name = key.replace("_agent_node", "").replace("run_", "").replace("_", " ").title().replace("Prep", " Prep")

                    # Log intermediate steps from the agent (ReAct agent)
                    if value.get("intermediate_steps"):
                        for action, observation in value["intermediate_steps"]:
                            st.session_state.agent_execution_log.append(f"➡️ **{agent_name} Action:** `{action.tool}({action.tool_input})`")
                            display_observation = str(observation)
                            if len(display_observation) > 100:
                                display_observation = display_observation[:97] + "..."
                            st.session_state.agent_execution_log.append(f"✅ **{agent_name} Observation:** `{display_observation}`")
                    
                    # Log characteristic results returned by each agent
                    if value.get("customer_info_result"):
                         st.session_state.agent_execution_log.append(f"📄 **{agent_name} Result:** Customer Info: {value['customer_info_result'].splitlines()[0]}...")
                         if value.get("customer_profile"):
                             st.session_state.agent_execution_log.append(f"👤 **{agent_name} Profile:** {value['customer_profile'].get('name', 'Unknown')} (ID: {value['customer_profile'].get('id', 'N/A')}) loaded.")
                    elif value.get("lead_info_result"):
                         st.session_state.agent_execution_log.append(f"📄 **{agent_name} Result:** Leads: {value['lead_info_result'].splitlines()[0]}...")
                    elif value.get("kb_info_result"):
                         st.session_state.agent_execution_log.append(f"📄 **{agent_name} Result:** KB Info: {value['kb_info_result'].splitlines()[0]}...")
                         if value.get("available_products_kb"):
                             st.session_state.agent_execution_log.append(f"📚 **{agent_name} Products:** Knowledge base content loaded for recommendations.")
                    elif value.get("recommendation_result"):
                         st.session_state.agent_execution_log.append(f"🌟 **{agent_name} Result:** Recommendations generated.")
                    elif key == "prepare_kb_query_for_recommendation":
                         st.session_state.agent_execution_log.append(f"📦 **{agent_name}:** Preparing KB query for recommendation.")

                # Log when the final response node finalizes an answer
                elif key == "final_response_node":
                    if value.get("final_response"):
                        st.session_state.agent_execution_log.append(f"✨ **Orchestrator: Finalizing Response**")
                
                # Check for errors produced by any node
                if value.get("error_message"): 
                    st.session_state.agent_execution_log.append(f"❌ **Error from {key}:** {value['error_message']}")
        
        # Get the final_response from the last state
        if last_state:
            full_response = last_state.get('final_response', "No final response generated.")
            if last_state.get('error_message') and not full_response:
                full_response = f"⚠️ An internal error occurred: {last_state['error_message']}"
            elif last_state.get('error_message'):
                full_response = f"⚠️ An internal error occurred: {last_state['error_message']}\n\n{full_response}"
        else:
            full_response = "No response was generated. Please try again."
            
    except Exception as e:
        # Handle critical workflow-level errors that occurred outside node-specific handlers
        full_response = f"An unexpected workflow error occurred: {e}. Please check the logs or try rephrasing your query."
        st.session_state.agent_execution_log.append(f"❌ **Critical Workflow Error**: {e}")
    
    end_time = time.time()
    response_time = end_time - start_time
    st.session_state.agent_execution_log.append(f"⏱️ **Response Time:** {response_time:.2f} seconds")

    return full_response


# --- Streamlit UI Layout ---
# Split layout into two columns: chat and log
col1, col2 = st.columns([0.7, 0.3])

with col1:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # User input box.
    # st.chat_input returns a value when the user presses Enter
    user_query = st.chat_input("Ask about customers, leads, or insurance policies...")
    
    # Process input when present
    if user_query:
        # Add the user's message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        # Execute the workflow and get an AI response
        with st.spinner("Processing your request..."):
            ai_response = get_response(user_query)
        
        # Add the AI's response to the chat history
        st.session_state.chat_history.append(AIMessage(content=ai_response))
        
        # Rerun to update the UI and clear the input box
        st.rerun()


with col2:
    st.header("🔍 Agent Execution Log")
    # Display logs in reverse order so the newest entries appear at the top
    if st.session_state.agent_execution_log:
        for log_entry in reversed(st.session_state.agent_execution_log): 
            st.markdown(log_entry)
    else:
        st.info("No agent activity yet. Ask a question to see the execution flow!")
    
    # Clear log button
    if st.button("🗑️ Clear Log", key="clear_log_button"):
        st.session_state.agent_execution_log = []
        st.rerun()

# --- Sidebar with example queries ---
st.sidebar.header("📝 Example Queries")
example_queries = {
    "Customer Queries": [
        "Find customer with email john@example.com",
        "Find customer John Smith",
        "Tell me about CUST003's policies."
    ],
    "Lead Queries": [
        "Show me qualified leads in Texas",
        "Find leads with score above 80 interested in auto insurance",
        "Are there any new leads interested in life insurance?"
    ],
    "Knowledge Queries": [
        "What is comprehensive auto insurance?",
        "What is an insurance deductible?",
        "What is a premium?",
        "Explain different types of life insurance.",
        "Tell me about car warranties?"
    ],
    "Recommendation Workflows": [
        "Find customer John Smith and recommend insurance products based on his profile",
        "Show me customer John Smith's current policies and recommend additional coverage options",
        "Recommend coverage for John Smith",
        "Find customer Emily Brown and recommend insurance products based on her profile",
        "Recommend products for non_existent@example.com",
    ],
    "General Fallback": [
        "Can you tell me a joke?",
        "Hello"
    ]
}

for category, queries in example_queries.items():
    st.sidebar.subheader(category)
    for query in queries:
        if st.sidebar.button(query, key=f"sidebar_query_{query}"):
            # When a sidebar sample-button is pressed, add the query to chat history
            st.session_state.chat_history.append(HumanMessage(content=query))
            
            # Process the query and obtain an AI response
            with st.spinner("Processing..."):
                ai_response = get_response(query)
            
            # Add the AI response to chat history
            st.session_state.chat_history.append(AIMessage(content=ai_response))
            st.rerun()  # Trigger a rerun to refresh the UI after the response

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.session_state.agent_execution_log = []
    st.rerun()