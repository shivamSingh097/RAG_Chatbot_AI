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
from langchain.agents import Tool, initialize_agent
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
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
    max_new_tokens=512,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Use a tool-based agent to decide between Wikipedia and local search
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
local_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

tools = [
    Tool(
        name="Local QA",
        func=local_qa.run,
        description="""Useful for answering questions about AI, Python, Economy, and general knowledge from local documents. 
        Input should be a standalone question."""
    ),
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="""Useful for answering a wide range of questions by searching on Wikipedia. 
        Input should be a standalone query."""
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
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
        with st.spinner("🔄 Thinking..."):
            try:
                # Use the agent to decide whether to use local QA or Wikipedia
                response = agent.run(user_question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                if st.session_state.current_chat_id is None:
                    st.session_state.current_chat_id = str(datetime.now().timestamp())
                save_chat_history(st.session_state.current_chat_id, user_question, response)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.exception(e) # This will display the full traceback for debugging
