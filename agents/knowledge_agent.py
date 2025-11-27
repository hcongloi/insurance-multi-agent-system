# agents/knowledge_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools.kb_tool import create_rag_knowledge_tool 
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_knowledge_agent(embeddings: GoogleGenerativeAIEmbeddings, vector_store: Chroma) -> AgentExecutor:
    """
    Creates and returns a knowledge agent capable of answering questions from an insurance knowledge base using RAG.
    It receives initialized embeddings and vector_store.
    """
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
    
    rag_tool_instance = create_rag_knowledge_tool(embeddings, vector_store)
    tools = [rag_tool_instance]

    kb_prompt_template = PromptTemplate.from_template(
        """You are an insurance knowledge expert. Your task is to answer questions about insurance products
        and policies based on the information available in your knowledge base.
        You have access to the following tool:
        {tools}
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (formulate a clear, concise query)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Use the 'query_knowledge_base_rag' tool to find relevant information.
        Always formulate a clear, concise question or topic for the tool.
        For example, if the user asks "What is term life insurance?", your tool input should be "term life insurance".
        If the user asks "What does comprehensive auto insurance cover?", your tool input should be "comprehensive auto insurance".
        
        If the tool indicates that no relevant information was found, respond politely that you cannot find the answer in the current knowledge base.
        Otherwise, summarize the information found in a clear and helpful manner to answer the user's question.
        Focus on providing direct answers based on the context provided by the tool.

        Begin!

        Question: {input}
        {agent_scratchpad}
        """
    )
    
    agent = create_react_agent(llm, tools, kb_prompt_template)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

# The if __name__ == "__main__": block for testing the agent will now be in langgraph_workflow.py or a dedicated test file