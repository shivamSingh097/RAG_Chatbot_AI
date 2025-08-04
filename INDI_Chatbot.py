import streamlit as st
import json
import os
from dotenv import load_dotenv
import sqlite3
import sys
from PIL import Image
from transformers import pipeline

# LangChain core components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load environment variables ---
load_dotenv()

# --- Load SERPER API Key from secrets ---
api_key = os.getenv("SERPER_API_KEY") or st.secrets["SERPER_API_KEY"]

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "loading" not in st.session_state:
    st.session_state.loading = False

# --- Theme Toggle ---
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# --- User Authentication Page ---
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

# --- Vector Store Setup (FAISS + HuggingFace) ---
loader = TextLoader("data.txt")
raw_docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# --- LLM Setup (HuggingFace FLAN-T5) ---
text_generation_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    device=-1,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- RetrievalQA Chain Setup ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# --- Web Search (Optional) ---
use_web_search = True
if use_web_search:
    search = GoogleSerperAPIWrapper()
    web_search_tool = Tool(
        name="Google Search",
        description="Search the web for current information",
        func=search.run
    )

# --- UI Setup ---
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

    st.markdown(f"👤 Logged in as: `{st.session_state.username}`")

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

# --- Custom CSS for Themes & Chat Bubbles ---
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
