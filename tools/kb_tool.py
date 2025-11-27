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
            print(f"Error during RAG query: {e}")
            return f"An error occurred while processing the knowledge base query: {e}. Please try again."
    return query_knowledge_base_rag