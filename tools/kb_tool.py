# tools/kb_tools.py
import re
import os
from typing import List, Dict, Any
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_rag_knowledge_tool(embeddings: GoogleGenerativeAIEmbeddings, vector_store: Chroma):
    @tool
    def query_knowledge_base_rag(query: str) -> str:
        """
        Retrieves relevant documents from the insurance knowledge base using RAG,
        then generates a concise answer using an LLM.
        Provide a specific question or topic, e.g., "What is auto insurance?",
        "Explain comprehensive coverage", "What is a premium?".
        """
        try:
            # Try RAG first
            relevant_docs = vector_store.similarity_search(query, k=5)
            
            if not relevant_docs:
                return f"No relevant information found in the knowledge base for '{query}'."

            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)
            
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an insurance expert. Answer the user's question ONLY based on the provided context.
                If the answer cannot be found in the context, state that you don't know or cannot provide the information.
                Provide a concise and helpful answer.
                
                Context:
                {context}"""),
                ("human", "{question}")
            ])

            chain = rag_prompt | llm
            response = chain.invoke({"context": context, "question": query}).content
            return response

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error during RAG query: {error_msg}")
            
            # Check if it's a quota error
            if "429" in error_msg or "quota" in error_msg.lower():
                return _fallback_knowledge_response(query)
            
            return f"An error occurred while processing the knowledge base query: {error_msg}. Please try again later."
    
    return query_knowledge_base_rag


def _fallback_knowledge_response(query: str) -> str:
    """
    Fallback responses when RAG is unavailable due to quota limits.
    Provides basic insurance knowledge without RAG.
    """
    query_lower = query.lower()
    
    fallback_kb = {
        "comprehensive": """Comprehensive auto insurance covers damage to your vehicle from non-collision events such as:
        - Theft or vandalism
        - Fire or explosion
        - Natural disasters (floods, hurricanes, hail)
        - Falling objects
        - Animal collisions (hitting a deer)
        
        It does NOT cover collision damage or liability.""",
        
        "collision": """Collision insurance covers damage to your vehicle from collisions with:
        - Other vehicles
        - Fixed objects (trees, poles, buildings)
        - Rollovers
        
        It covers repair costs regardless of fault.""",
        
        "life insurance": """Life insurance provides financial protection to your beneficiaries when you pass away. Types include:
        - Term Life: Coverage for a specific period (10, 20, 30 years)
        - Whole Life: Permanent coverage with cash value component
        - Universal Life: Flexible premiums and death benefits""",
        
        "premium": """An insurance premium is the amount you pay (usually monthly or annually) to keep your insurance policy active. Factors affecting premiums:
        - Coverage amount
        - Deductible chosen
        - Risk factors (age, driving record, health)
        - Location""",
        
        "deductible": """A deductible is the amount you pay out-of-pocket before insurance coverage kicks in. Examples:
        - $500 deductible: You pay first $500, insurance pays the rest
        - Higher deductible = Lower premium
        - Lower deductible = Higher premium""",
        
        "liability": """Liability insurance covers damages you cause to others:
        - Bodily injury liability: Medical costs for injured parties
        - Property damage liability: Repair costs for damaged property
        - Legal defense costs if sued
        
        Required by law in most states.""",
        
        "health insurance": """Health insurance helps pay for medical expenses including:
        - Doctor visits
        - Hospital stays
        - Prescription medications
        - Preventive care
        - Emergency services
        
        Types: HMO, PPO, EPO, POS""",
        
        "auto insurance": """Auto insurance protects you financially in vehicle-related incidents. Main types:
        - Liability: Covers damage you cause to others (required)
        - Collision: Covers your vehicle damage in accidents
        - Comprehensive: Covers non-collision damage
        - Uninsured Motorist: Protects you from uninsured drivers"""
    }
    
    # Find matching topic
    for keyword, answer in fallback_kb.items():
        if keyword in query_lower:
            return f"⚠️ *Using cached knowledge (RAG temporarily unavailable)*\n\n{answer}\n\n📚 For more detailed information, please try again later when the knowledge base is available."
    
    return f"""⚠️ The knowledge base is temporarily unavailable due to API quota limits.
    
I can provide basic information about:
- Auto insurance (comprehensive, collision, liability)
- Life insurance (term, whole, universal)
- Health insurance
- Common terms (premium, deductible, coverage)

Please ask about one of these topics, or try again later for more detailed information."""