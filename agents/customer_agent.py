# agents/customer_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent 
from langchain_core.prompts import PromptTemplate
from tools.crm_tool import get_customer_info 
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME

def create_customer_agent() -> AgentExecutor:
    """
    Creates and returns a customer agent capable of retrieving customer information.
    """
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
    tools = [get_customer_info]

    customer_prompt_template = PromptTemplate.from_template(
        """You are a helpful customer service agent for an insurance company.
        Your primary goal is to retrieve customer information based on user queries.
        You have access to the following tools:
        {tools}
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (should be a customer ID, email, name, or policy ID)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Use the 'get_customer_info' tool to find customer details.
        
        When using the tool, extract the customer ID, email, name, or policy ID from the user query.
        For example:
        - "Find customer with email john@example.com" -> tool input: "john@example.com"
        - "Show me info for customer CUST001" -> tool input: "CUST001"
        - "Who owns policy AUTO-001?" -> tool input: "AUTO-001"
        - "Get details for Jane Doe" -> tool input: "Jane Doe"

        If the tool returns an empty dictionary, it means the customer was not found.
        In that case, respond politely that the customer could not be found and ask for clarification.

        If the customer is found, present their information in a clear, human-readable format,
        categorizing details like "Contact Information", "Policies", and "History".
        Do not output raw JSON directly to the user.

        Begin!

        Question: {input}
        {agent_scratchpad}
        """
    )
    
    agent = create_react_agent(llm, tools, customer_prompt_template)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

if __name__ == "__main__":
    customer_agent = create_customer_agent()

    print("\n--- Test Customer Agent ---")

    queries = [
        "Find customer with email john@example.com",
        "Show me information for CUST002",
        "Get details for policy AUTO-001",
        "Who is Jane Doe?",
        "Find customer named Robert Johnson",
        "Retrieve customer with ID CUST005",
        "Find customer with email non_existent@example.com",
        "What about customer 999?",
        "I need info on Policy NON-EXISTENT-001"
    ]

    for query in queries:
        print(f"\nQUERY: {query}")
        try:
            response = customer_agent.invoke({"input": query})
            print(f"RESPONSE:\n{response['output']}")
        except Exception as e:
            print(f"ERROR processing query: {e}")