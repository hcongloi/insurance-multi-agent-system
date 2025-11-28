# main.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time

# Import create_multi_agent_workflow from langgraph_workflow.py
from langgraph_workflow import create_multi_agent_workflow

# Use st.cache_resource so the LangGraph app is initialized only once.
@st.cache_resource
def get_langgraph_app():
    return create_multi_agent_workflow()

app = get_langgraph_app()

# Configure the Streamlit page
st.set_page_config(
    page_title="Insurance AI Assistant", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Show sidebar toggle button */
    button[kind="header"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Chat container */
    .chat-container {
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        min-height: 500px;
        max-height: 650px;
        overflow-y: auto;
    }
    
    .chat-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #87b5ff;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    /* Custom chat messages */
    .stChatMessage {
        background: transparent !important;
        padding: 1rem 0 !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: transparent !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: transparent !important;
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
        margin-bottom: 1.5rem;
    }
    
    .stats-number {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .stats-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Execution log */
    .log-container {
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        max-height: 450px;
        overflow-y: auto;
    }
    
    .log-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #87b5ff;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .log-entry {
        background: linear-gradient(to right, #f8fafc 0%, #ffffff 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-radius: 8px;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
        color: #1e293b;
    }
    
    .log-entry:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .log-empty {
        text-align: center;
        padding: 3rem 1rem;
        color: #64748b;
    }
    
    .log-empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        width: 300px !important;
    }
    
    .sidebar-title {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        text-align: center;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 10px;
        font-weight: 600;
        color: #1e293b !important;
        padding: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f1f5f9;
        border-color: #667eea;
    }
    
    /* Chat input */
    .stChatInput {
        border-radius: 12px;
    }
    
    .stChatInput > div {
        border-radius: 12px;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_execution_log" not in st.session_state:
    st.session_state.agent_execution_log = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

def get_response(user_query: str) -> str:
    """
    Execute the multi-agent workflow and stream results.
    """
    st.session_state.agent_execution_log = []
    st.session_state.total_queries += 1
    
    inputs = {
        "input": user_query,
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
        "router_decision": ""
    }
    
    full_response = ""
    last_state = None
    start_time = time.time()
    
    try:
        for s in app.stream(inputs, stream_mode="updates"):
            for key, value in s.items():
                if isinstance(value, dict):
                    if last_state is None:
                        last_state = value.copy()
                    else:
                        last_state.update(value)
                
                if key == "router_node":
                    router_decision = value.get("router_decision") 
                    if router_decision:
                        st.session_state.agent_execution_log.append(
                            f"🔄 **Routing Decision:** `{router_decision}`"
                        )
                
                elif key == "set_recommendation_flag":
                    st.session_state.agent_execution_log.append(
                        "🚩 **Flag Set:** Recommendation flow activated"
                    )
                
                elif key.endswith("_agent_node") or key == "run_recommendation_node" or key == "prepare_kb_query_for_recommendation":
                    agent_name = key.replace("_agent_node", "").replace("run_", "").replace("_", " ").title()

                    if value.get("intermediate_steps"):
                        for action, observation in value["intermediate_steps"]:
                            st.session_state.agent_execution_log.append(
                                f"🔧 **{agent_name}:** `{action.tool}({action.tool_input})`"
                            )
                            display_observation = str(observation)
                            if len(display_observation) > 100:
                                display_observation = display_observation[:97] + "..."
                            st.session_state.agent_execution_log.append(
                                f"✅ **Result:** {display_observation}"
                            )
                    
                    if value.get("customer_info_result"):
                        st.session_state.agent_execution_log.append(
                            f"👤 **Customer Info:** Found customer data"
                        )
                    elif value.get("lead_info_result"):
                        st.session_state.agent_execution_log.append(
                            f"📊 **Leads Info:** Retrieved lead information"
                        )
                    elif value.get("kb_info_result"):
                        st.session_state.agent_execution_log.append(
                            f"📚 **Knowledge Base:** Retrieved relevant information"
                        )
                    elif value.get("recommendation_result"):
                        st.session_state.agent_execution_log.append(
                            "🎯 **Recommendations:** Generated personalized recommendations"
                        )

                elif key == "final_response_node":
                    if value.get("final_response"):
                        st.session_state.agent_execution_log.append(
                            "✨ **Response:** Finalized and ready"
                        )
                
                if value.get("error_message"): 
                    st.session_state.agent_execution_log.append(
                        f"❌ **Error:** {value['error_message']}"
                    )
        
        if last_state:
            full_response = last_state.get('final_response', "No final response generated.")
            if last_state.get('error_message') and not full_response:
                full_response = f"⚠️ An internal error occurred: {last_state['error_message']}"
            elif last_state.get('error_message'):
                full_response = f"⚠️ An internal error occurred: {last_state['error_message']}\n\n{full_response}"
        else:
            full_response = "No response was generated. Please try again."
            
    except Exception as e:
        full_response = f"An unexpected error occurred: {e}"
        st.session_state.agent_execution_log.append(f"❌ **Critical Error:** {e}")
    
    end_time = time.time()
    response_time = end_time - start_time
    st.session_state.agent_execution_log.append(
        f"⏱️ **Completed in:** {response_time:.2f} seconds"
    )

    return full_response


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛡️ Insurance AI Assistant</h1>
    <p class="subtitle">Powered by LangChain 1.0.5 & Google Gemini 1.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# --- Main Layout ---
col1, col2 = st.columns([0.65, 0.35], gap="large")

with col1:
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">💬 Conversation</div>', unsafe_allow_html=True)
    
    # Display chat history
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #64748b;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">💭</div>
            <div style="font-size: 1.1rem; font-weight: 500;">Start a conversation</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Ask me about customers, leads, policies, or get recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)
    
    st.markdown('</div>', unsafe_allow_html=True)

# User input (outside columns for full width)
user_query = st.chat_input("💭 Type your message here...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.spinner("🔄 Processing your request..."):
        ai_response = get_response(user_query)
    
    st.session_state.chat_history.append(AIMessage(content=ai_response))
    st.rerun()


with col2:
    # Stats card
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{st.session_state.total_queries}</div>
        <div class="stats-label">QUERIES PROCESSED</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Execution log
    # st.markdown('<div class="log-container">', unsafe_allow_html=True)
    st.markdown('<div class="log-header">🔍 Execution Log</div>', unsafe_allow_html=True)
    
    if st.session_state.agent_execution_log:
        for log_entry in reversed(st.session_state.agent_execution_log): 
            st.markdown(f'<div class="log-entry">{log_entry}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="log-empty">
            <div class="log-empty-icon">📋</div>
            <div>No activity yet</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">Ask a question to see the execution flow</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state.agent_execution_log = []
            st.rerun()
    with col_btn2:
        if st.button("🔄 Reset All", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.agent_execution_log = []
            st.session_state.total_queries = 0
            st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">📝 Quick Examples</div>', unsafe_allow_html=True)
    
    example_queries = {
        "👤 Customer Queries": [
            "Find customer with email john@example.com",
            "Find customer John Smith",
            "Tell me about CUST003's policies"
        ],
        "📊 Lead Queries": [
            "Show me qualified leads in Texas",
            "Find leads with score above 80",
            "New leads interested in life insurance?"
        ],
        "📚 Knowledge Base": [
            "What is comprehensive auto insurance?",
            "Explain insurance deductibles",
            "Types of life insurance"
        ],
        "🎯 Recommendations": [
            "Recommend products for John Smith",
            "Show Emily Brown's coverage options",
            "Suggest insurance for CUST003"
        ],
        "💬 General": [
            "Hello",
            "How can you help me?"
        ]
    }

    for category, queries in example_queries.items():
        with st.expander(category, expanded=False):
            for query in queries:
                if st.button(query, key=f"sidebar_{query}", use_container_width=True):
                    st.session_state.chat_history.append(HumanMessage(content=query))
                    
                    with st.spinner("Processing..."):
                        ai_response = get_response(query)
                    
                    st.session_state.chat_history.append(AIMessage(content=ai_response))
                    st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Info section
    with st.expander("ℹ️ About This Assistant", expanded=False):
        st.markdown("""
        **Multi-Agent AI System**
        
        Specialized agents working together:
        
        🔍 **Customer Agent**  
        Search and retrieve customer data
        
        📈 **Lead Agent**  
        Analyze and filter leads
        
        📖 **Knowledge Agent**  
        Query insurance information
        
        🎯 **Recommendation Agent**  
        Generate personalized suggestions
        
        ---
        
        Built with **LangGraph** & **Gemini AI**
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.agent_execution_log = []
        st.rerun()