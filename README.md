# Multi-Agent Insurance Assistant

A small multi-agent demo that combines a LangGraph orchestrator and specialized agents (Customer, Lead, Knowledge + RAG) to answer queries, lookup CRM data, search lead lists, and recommend insurance products. Built with LangChain-style agents, Google Gemini LLMs (via langchain-google-genai), and a Chroma vector store for RAG over a local markdown knowledge base.

This README covers quick setup, how to run the app locally (Streamlit), and where to find the main components. For a detailed per-file functional description see `docs/FUNCTIONAL_SPEC.md`.

---

## Quick overview

- Entry point (UI): `main.py` — Streamlit app which uses a compiled LangGraph workflow to process user queries.
- Orchestrator / workflow: `langgraph_workflow.py` — router node + specialized agent nodes + the orchestration logic.
- Agents: `agents/` — `customer_agent.py`, `lead_agent.py`, `knowledge_agent.py` (wrap individual AgentExecutors for specific tasks).
- Tools: `tools/` — `crm_tool.py` (data access for customers & leads), `kb_tool.py` (RAG wrapper), `recommendation_tool.py` (simple recommendation logic).
- RAG utils: `utils/rag_pipeline.py` — ingestion, splitting, Chroma vector store creation & retrieval.
- Data: `data/*` — `customers.json`, `leads.json`, `insurance_kb.md` (knowledge base used for RAG).

---

## Setup (Windows)

1. Create and activate a Python virtual environment (the repository contains an example `.venv` but you may create your own):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

2. Install dependencies (use your venv):

```cmd
pip install -r requirements.txt
```

3. Add your Google API credentials in `.env` in the repository root. Example (already present in this repo as a demo but typically you will provide your own):

```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GEMINI_MODEL_NAME=gemini-2.5-flash-lite
```

4. If the Chroma DB is not present, the workflow will attempt to ingest `data/insurance_kb.md` automatically and create the vectorstore under `vectorstore/chroma_db`.
   If it is still not created, please create it manually by running and create the vectorstore under same folder.

```cmd
python -m utils/rag_pipeline.py
```

---

## Run the app (Streamlit)

```cmd
streamlit run main.py
```

Open http://localhost:8501 and use the chat input or the example sidebar queries.

---

## Examples (what to ask)

- Customer queries: "Find customer with email john@example.com", "Tell me about CUST003's policies"
- Lead queries: "Show me qualified leads in Texas", "Find leads with score above 80 interested in auto insurance"
- Knowledge queries (RAG): "What is comprehensive auto insurance?", "What is a deductible?"
- Recommendation workflows: "Find customer John Smith and recommend insurance products based on his profile"

---

## Where to look next

- Implementation and orchestration details: `langgraph_workflow.py`
- Tools (data + RAG) and helpers: `tools/` and `utils/rag_pipeline.py`
- To add or update knowledge content, edit `data/insurance_kb.md` and let the ingestion recreate the Chroma DB.

If you want, I can also add inline docstrings, type hints, or unit tests for critical modules (CRM tools, RAG ingestion, recommendation logic). Tell me what you'd like next. ✅
