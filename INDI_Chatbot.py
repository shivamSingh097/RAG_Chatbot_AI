import os
import pickle
import json
import streamlit as st
from PIL import Image
from huggingface_hub import login
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from datetime import datetime

# ===================== API Keys & Configuration =====================
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", st.secrets.get("HUGGINGFACEHUB_API_TOKEN"))
if not HF_TOKEN:
    st.error("❌ Hugging Face token is missing. Add it to .streamlit/secrets.toml")
    st.stop()
login(HF_TOKEN)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", st.secrets.get("SERPER_API_KEY"))

# ===================== Check FAISS Index =====================
FAISS_INDEX_PATH = "faiss_index.pkl"
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("❌ FAISS index not found. Run ingest.py first to create it.")
    st.stop()

with open(FAISS_INDEX_PATH, "rb") as f:
    vector_db = pickle.load(f)

# ===================== Session State & Chat History Management =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "username" not in st.session_state:
    st.session_state.username = "Guest"

CHAT_HISTORY_FILE = "chat_history.json"

def save_chat_history(chat_id, user_message, ai_message):
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            history = json.load(f)
    
    chat_found = False
    for chat in history:
        if chat["id"] == chat_id:
            chat["conversation"].append({"user": user_message, "ai": ai_message})
            chat["last_updated"] = datetime.now().isoformat()
            chat_found = True
            break
            
    if not chat_found:
        history.append({
            "id": chat_id,
            "title": user_message[:30] + "...",
            "last_updated": datetime.now().isoformat(),
            "conversation": [{"user": user_message, "ai": ai_message}]
        })
        
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
        
def load_chat(chat_id):
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            history = json.load(f)
        for chat in history:
            if chat["id"] == chat_id:
                st.session_state.messages = []
                for msg in chat["conversation"]:
                    st.session_state.messages.append({"role": "user", "content": msg["user"]})
                    st.session_state.messages.append({"role": "assistant", "content": msg["ai"]})
                st.session_state.current_chat_id = chat_id
                st.session_state.chat_history = chat["conversation"]
                st.rerun()

# ===================== Authentication =====================
def login_page():
    # Your existing login_page function...
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

if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    login_page()
    st.stop()

# ===================== LLM & Chains =====================
text_generation_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

use_web_search = bool(SERPER_API_KEY)
if use_web_search:
    search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    web_search_tool = Tool(
        name="Google Search",
        description="Fetch real-time web search results",
        func=search.run
    )

# ===================== Sidebar =====================
st.set_page_config(page_title="INDIBOT AI", layout="wide")
with st.sidebar:
    st.header("👤 Profile")
    st.markdown(f"**Logged in as:** `{st.session_state.username}`")
    if st.button("Logout", key="logout_btn"):
        st.session_state.user_logged_in = False
        st.session_state.username = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")
    
    st.header("💬 Recent Chats")
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            all_chats = json.load(f)
        for chat in all_chats:
            if st.button(f"📄 {chat['title']}", key=f"chat_{chat['id']}"):
                load_chat(chat['id'])
    
    st.markdown("---")
    if st.button("🆕 New Chat", key="new_chat_btn"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.rerun()

# ===================== Main UI =====================
st.title("🧠 IndiBot AI")
st.write("Ask me anything about AI, Python, Economy, General Knowledge or Live Web Search! ✨")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("🎤 Ask your question..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
        
    found_in_history = False
    
    # Check if the question is in the current chat history
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i]["content"] == user_question:
            if i + 1 < len(st.session_state.messages):
                ai_answer = st.session_state.messages[i+1]["content"]
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                with st.chat_message("assistant"):
                    st.markdown("**(Found in chat history):** " + ai_answer)
                found_in_history = True
                break

    if not found_in_history:
        with st.spinner("🔄 Checking local documents..."):
            try:
                local_answer = qa_chain.run(user_question)
                st.session_state.messages.append({"role": "assistant", "content": local_answer})
                with st.chat_message("assistant"):
                    st.markdown(local_answer)
                
                # Save the new conversation to the history file
                if st.session_state.current_chat_id is None:
                    st.session_state.current_chat_id = str(datetime.now().timestamp())
                save_chat_history(st.session_state.current_chat_id, user_question, local_answer)
                
            except Exception as e:
                st.error(f"❌ Local QA failed: {e}")
                
        # ============ Run Web Search if API available and local QA failed or was unhelpful ============
        if use_web_search and "i don't know" in local_answer.lower():
            with st.spinner("🌐 Performing web search..."):
                try:
                    web_result = web_search_tool.run(user_question)
                    final_answer = f"### 🌐 Web Search Result\n{web_result}"
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                except Exception as e:
                    st.warning("⚠️ Web search failed.")
                    st.error(str(e))
    st.rerun()
