# agents/lead_agent.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools.crm_tool import search_leads 
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME

def create_lead_agent() -> AgentExecutor:
    """
    Creates and returns a lead agent capable of searching for qualified leads.
    """
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
    tools = [search_leads]

    lead_prompt_template = PromptTemplate.from_template(
        """You are a lead qualification agent for an insurance company.
        Your goal is to identify potential leads based on specific criteria provided by the user.
        You have access to the following tools:
        {tools}

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (formulate a clear, concise query)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Use the 'search_leads' tool to filter leads.
        
        The user will provide criteria like minimum score, interest areas (e.g., "auto", "life"),
        geographic area, or status (e.g., "New", "Qualified").
        Always extract these criteria and form a JSON dictionary for the 'search_leads' tool.
        
        Example tool input format: {{"score_min": 70, "interest": "auto", "status": "New"}}
        If no specific criteria are mentioned, try to infer general search or ask for clarification.

        If the tool returns an empty list, respond politely that no leads matching the criteria were found.
        If leads are found, list them clearly with their ID, name, score, interest, area, and status.
        Format the output clearly, do not just dump JSON.

        Begin!

        Question: {input}
        {agent_scratchpad}
        """
    )
    
    agent = create_react_agent(llm, tools, lead_prompt_template)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

if __name__ == "__main__":
    lead_agent = create_lead_agent()

    print("\n--- Test Lead Agent ---")

    queries = [
        "Find all new leads interested in auto insurance with a score above 80",
        "Show me qualified leads in Texas",
        "Are there any new leads named John Doe?",
        "Find leads with score below 60",
        "List all leads interested in property insurance",
        "Show me all leads in California",
        "Who is lead Sarah Connor?",
        "Find leads with score >= 90"
    ]

    for query in queries:
        print(f"\nQUERY: {query}")
        try:
            response = lead_agent.invoke({"input": query})
            print(f"RESPONSE:\n{response['output']}")
        except Exception as e:
            print(f"ERROR processing query: {e}")