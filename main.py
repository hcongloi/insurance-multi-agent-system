# main.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time  # Thêm để mô phỏng loading time hoặc các animations nhỏ

# Import create_multi_agent_workflow từ file langgraph_workflow.py
from langgraph_workflow import create_multi_agent_workflow

# Sử dụng st.cache_resource để khởi tạo LangGraph app một lần duy nhất.
# Điều này rất quan trọng để tránh khởi tạo lại các model và vectorstore trên mỗi lần rerun của Streamlit.
@st.cache_resource
def get_langgraph_app():
    return create_multi_agent_workflow()

app = get_langgraph_app()

# Cấu hình trang Streamlit
st.set_page_config(page_title="Multi-Agent Insurance Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Insurance Assistant")
st.caption("Powered by LangChain 1.0.5 & Google Gemini 1.5 Flash")

# Khởi tạo session state nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_execution_log" not in st.session_state:
    st.session_state.agent_execution_log = []

def get_response(user_query: str) -> str:
    """
    Thực thi workflow đa tác tử và stream kết quả.
    Ghi log các bước trung gian vào st.session_state.agent_execution_log.
    Trả về phản hồi cuối cùng dưới dạng chuỗi.
    """
    st.session_state.agent_execution_log = []  # Đặt lại log cho mỗi truy vấn mới
    
    # Tạo trạng thái ban đầu cho workflow. Quan trọng là phải là một dict mới mỗi lần.
    inputs = {
        "input": user_query,
        "chat_history": [],  # Truyền chat_history rỗng vì AgentState có Annotated[List[BaseMessage], operator.add]
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
        "router_decision": ""  # Khởi tạo router_decision
    }
    
    full_response = ""
    last_state = None  # Thêm biến để lưu state cuối cùng
    start_time = time.time()  # Bắt đầu tính thời gian phản hồi
    
    try:
        # Stream các bước thực thi để theo dõi
        for s in app.stream(inputs, stream_mode="updates"):
            # stream_mode="updates" trả về dict với key là tên node và value là state updates
            for key, value in s.items():
                # Lưu state cuối cùng từ mỗi update
                if isinstance(value, dict):
                    if last_state is None:
                        last_state = value.copy()
                    else:
                        last_state.update(value)
                
                # Ghi log cho router node
                if key == "router_node":
                    router_decision = value.get("router_decision") 
                    if router_decision:
                        st.session_state.agent_execution_log.append(f"🔄 **Orchestrator Routing:** Decided to use `{router_decision}`")
                
                # Ghi log cho flag khuyến nghị
                elif key == "set_recommendation_flag":
                    st.session_state.agent_execution_log.append(f"🚩 **Orchestrator Flag:** `is_recommendation_flow` set to `True`.")
                
                # Ghi log cho các agent và node khác
                elif key.endswith("_agent_node") or key == "run_recommendation_node" or key == "prepare_kb_query_for_recommendation":
                    agent_name = key.replace("_agent_node", "").replace("run_", "").replace("_", " ").title().replace("Prep", " Prep")

                    # Ghi log các bước trung gian của agent (React agent)
                    if value.get("intermediate_steps"):
                        for action, observation in value["intermediate_steps"]:
                            st.session_state.agent_execution_log.append(f"➡️ **{agent_name} Action:** `{action.tool}({action.tool_input})`")
                            display_observation = str(observation)
                            if len(display_observation) > 100:
                                display_observation = display_observation[:97] + "..."
                            st.session_state.agent_execution_log.append(f"✅ **{agent_name} Observation:** `{display_observation}`")
                    
                    # Ghi log kết quả đặc trưng của từng agent
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

                # Ghi log khi response cuối cùng được finalize
                elif key == "final_response_node":
                    if value.get("final_response"):
                        st.session_state.agent_execution_log.append(f"✨ **Orchestrator: Finalizing Response**")
                
                # Kiểm tra lỗi từ bất kỳ node nào
                if value.get("error_message"): 
                    st.session_state.agent_execution_log.append(f"❌ **Error from {key}:** {value['error_message']}")
        
        # Lấy final_response từ state cuối cùng
        if last_state:
            full_response = last_state.get('final_response', "No final response generated.")
            if last_state.get('error_message') and not full_response:
                full_response = f"⚠️ An internal error occurred: {last_state['error_message']}"
            elif last_state.get('error_message'):
                full_response = f"⚠️ An internal error occurred: {last_state['error_message']}\n\n{full_response}"
        else:
            full_response = "No response was generated. Please try again."
            
    except Exception as e:
        # Xử lý các lỗi nghiêm trọng xảy ra ngoài các node cụ thể
        full_response = f"An unexpected workflow error occurred: {e}. Please check the logs or try rephrasing your query."
        st.session_state.agent_execution_log.append(f"❌ **Critical Workflow Error**: {e}")
    
    end_time = time.time()
    response_time = end_time - start_time
    st.session_state.agent_execution_log.append(f"⏱️ **Response Time:** {response_time:.2f} seconds")

    return full_response


# --- Streamlit UI Layout ---
# Chia layout thành hai cột: chat và log
col1, col2 = st.columns([0.7, 0.3])

with col1:
    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Ô nhập liệu cho người dùng.
    # st.chat_input trả về giá trị khi user nhấn Enter
    user_query = st.chat_input("Ask about customers, leads, or insurance policies...")
    
    # Xử lý input nếu có
    if user_query:
        # Thêm tin nhắn của người dùng vào lịch sử chat
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        # Xử lý truy vấn và lấy phản hồi AI
        with st.spinner("Processing your request..."):
            ai_response = get_response(user_query)
        
        # Thêm phản hồi AI vào lịch sử chat
        st.session_state.chat_history.append(AIMessage(content=ai_response))
        
        # Rerun để cập nhật UI và clear input
        st.rerun()


with col2:
    st.header("🔍 Agent Execution Log")
    # Hiển thị log ngược lại để các log mới nhất nằm trên cùng
    if st.session_state.agent_execution_log:
        for log_entry in reversed(st.session_state.agent_execution_log): 
            st.markdown(log_entry)
    else:
        st.info("No agent activity yet. Ask a question to see the execution flow!")
    
    # Nút xóa log
    if st.button("🗑️ Clear Log", key="clear_log_button"):
        st.session_state.agent_execution_log = []
        st.rerun()

# --- Thanh sidebar với các câu truy vấn ví dụ ---
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
            # Khi một nút sidebar được nhấn, thêm truy vấn vào lịch sử chat
            st.session_state.chat_history.append(HumanMessage(content=query))
            
            # Xử lý truy vấn và lấy phản hồi AI
            with st.spinner("Processing..."):
                ai_response = get_response(query)
            
            # Thêm phản hồi AI vào lịch sử chat
            st.session_state.chat_history.append(AIMessage(content=ai_response))
            st.rerun()  # Gọi rerun để cập nhật UI sau khi có phản hồi

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.session_state.agent_execution_log = []
    st.rerun()