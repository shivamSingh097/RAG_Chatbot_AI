import os
import pickle
import json
import streamlit as st
from PIL import Image
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain_community.llms import HuggingFacePipeline

# ===================== API Key =====================
# Load environment variables (SERPER_API_KEY etc.)
api_key = st.secrets["SERPER_API_KEY"]

# ===================== Check FAISS Index =====================
FAISS_INDEX_PATH = "faiss_index.pkl"
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("❌ FAISS index not found. Run ingest.py first to create it.")
    st.stop()

with open(FAISS_INDEX_PATH, "rb") as f:
    vector_db = pickle.load(f)

# ===================== Session State =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "loading" not in st.session_state:
    st.session_state.loading = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ===================== Authentication =====================
def login_page():
    st.title("🔐 Welcome to INDIBOT")
    st.markdown("---")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                try:
                    with open("users.json", 'r') as f:
                        users = json.load(f)
                    user_found = any(u['username'] == username and u['password'] == password for u in users)
                    if user_found:
                        st.session_state.user_logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except FileNotFoundError:
                    st.error("No registered users found. Please register first.")

    with tab2:
        st.subheader("Create a new account")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_email = st.text_input("Email ID", key="reg_email")
        if st.button("Register", key="register_button"):
            if not reg_username or not reg_password or not reg_email:
                st.error("Username, password, and email are compulsory.")
            else:
                if not os.path.exists("users.json"):
                    with open("users.json", "w") as f:
                        json.dump([], f)
                with open("users.json", 'r') as f:
                    users = json.load(f)
                if any(u['username'] == reg_username for u in users):
                    st.error("Username already exists. Please choose another.")
                elif any(u['email'] == reg_email for u in users):
                    st.error("Email ID already registered. Please use another.")
                else:
                    user_data = {"username": reg_username, "password": reg_password, "email": reg_email}
                    users.append(user_data)
                    with open("users.json", 'w') as f:
                        json.dump(users, f, indent=4)
                    st.success("Registration successful! You can now log in.")

if "user_logged_in" not in st.session_state:
    login_page()
    st.stop()

# ===================== LLM =====================
text_generation_pipeline = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    max_new_tokens=512,
    temperature=0.7,
    device_map="auto"   # uses CPU on Streamlit Cloud
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# ===================== Retrieval QA =====================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# ===================== Web Search =====================
use_web_search = bool(os.environ.get("SERPER_API_KEY"))
if use_web_search:
    search = GoogleSerperAPIWrapper()
    web_search_tool = Tool(
        name="Google Search",
        description="Search the web for current information",
        func=search.run
    )

# ===================== Sidebar =====================
st.set_page_config(page_title="INDI_Chatbot", layout="wide")
with st.sidebar:
    try:
        logo = Image.open("INDIBOT.png")
        st.image(logo, width=100)
    except:
        st.markdown("⚠️ Add 'INDIBOT.png' in your project folder.")
    st.title("📚 Chat History")
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"- 💬 {msg['query'][:30]}...")
    if st.button("🆕 New Chat"):
        st.session_state.chat_history.clear()
        st.rerun()
    st.markdown("---")
    if st.button("🌓 Toggle Theme"):
        toggle_theme()
        st.rerun()
    st.markdown(f"👤 Logged in as: {st.session_state.username}")

# ===================== Main UI =====================
st.title("🧠 INDIBOT - Vocal for Local 🇮🇳")
st.markdown("Ask me anything about AI, Python, Economy or General Knowledge! ✨")

user_question = st.text_input("🎤 Ask your question:", placeholder="Type your query here...")
if st.button("Get Answer") or user_question:
    if user_question:
        st.session_state.loading = True
        with st.spinner("🔄 Thinking..."):
            try:
                local_answer = qa_chain.run(user_question)
                st.session_state.loading = False
                st.chat_message("user", avatar="👤").write(user_question)
                st.chat_message("ai", avatar="🤖").write(local_answer)
                st.session_state.chat_history.append({
                    "user": st.session_state.username,
                    "query": user_question,
                    "answer": local_answer
                })
            except Exception as e:
                st.session_state.loading = False
                st.error(f"❌ Local QA failed: {e}")
        if use_web_search:
            try:
                web_result = web_search_tool.run(user_question)
                st.markdown("### 🌐 Web Search Result")
                st.info(web_result)
            except Exception as e:
                st.warning("⚠️ Web search failed. Check SERPER_API_KEY or internet.")
                st.error(str(e))

# ===================== Theme CSS =====================
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #121212; color: #f0f0f0; }
        .stTextInput, .stTextArea { background-color: #333 !important; color: white; }
        .css-1cpxqw2 { background-color: #1f1f1f !important; }
        .element-container:has(.stSpinner) .stSpinner { color: #00FFAA; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #ffffff; color: #000000; }
        </style>
    """, unsafe_allow_html=True)
