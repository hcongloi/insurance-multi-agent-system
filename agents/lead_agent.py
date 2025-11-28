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

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        When using the search_leads tool, provide the criteria as a JSON string.
        
        CRITICAL: Action Input must be a valid JSON string enclosed in quotes.
        
        ✅ CORRECT examples:
        Action Input: {{"status": "Qualified", "area": "Texas"}}
        Action Input: {{"score_min": 80, "interest": "auto"}}
        Action Input: {{"name": "John", "status": "New"}}
        Action Input: {{"score_min": 70}}
        Action Input: {{}}
        
        ❌ WRONG examples:
        Action Input: status=Qualified, area=Texas
        Action Input: Qualified leads in Texas
        
        Available criteria keys:
        - score_min: minimum lead score (integer)
        - interest: interest keyword (e.g., "auto", "life", "home")
        - area: geographic area (e.g., "Texas", "California")
        - status: lead status (e.g., "New", "Contacted", "Qualified", "Lost")
        - name: part of lead's name
        
        Extract criteria from the user's query and format as JSON.
        If no criteria mentioned, use empty JSON: {{}}
        
        When you receive results:
        - If empty list: say no leads were found matching the criteria
        - If leads found: format them clearly with ID, name, score, interest, area, status, and contact info
        
        Present results in a human-readable format, not raw JSON.

        Begin!

        Question: {input}
        {agent_scratchpad}
        """
    )
    
    agent = create_react_agent(llm, tools, lead_prompt_template)
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )
    return executor


if __name__ == "__main__":
    lead_agent = create_lead_agent()

    print("\n" + "="*80)
    print("Testing Lead Agent with JSON String Input")
    print("="*80)

    queries = [
        "Find all new leads interested in auto insurance with a score above 80",
        "Show me qualified leads in Texas",
        "Are there any new leads named John?",
        "Find leads with score below 60",
        "List all leads in California",
        "Find leads with score >= 90"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print("="*80)
        try:
            response = lead_agent.invoke({"input": query})
            print(f"\n✅ RESPONSE:\n{response['output']}")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")