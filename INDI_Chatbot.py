import os
import sys
import json
import pickle
from datetime import datetime
import streamlit as st
from PIL import Image

# ==== Fix for SQLite (ChromaDB / FAISS) on Streamlit Cloud ====
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ==== NLP libraries ====
import spacy
from textblob import TextBlob
from difflib import get_close_matches

from huggingface_hub import login
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun

# ================== API KEYS ==================
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", st.secrets.get("HUGGINGFACEHUB_API_TOKEN"))
if not HF_TOKEN:
    st.error("❌ Hugging Face token is missing in .streamlit/secrets.toml")
    st.stop()
login(HF_TOKEN)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", st.secrets.get("SERPER_API_KEY"))

# ================== Load Vector DB ==================
FAISS_INDEX_PATH = "faiss_index.pkl"
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("❌ FAISS index not found. Run ingest.py first.")
    st.stop()

with open(FAISS_INDEX_PATH, "rb") as f:
    vector_db = pickle.load(f)

# ================== NLP Initialization ==================
nlp = spacy.load("en_core_web_sm")

# ================== Session & Chat ==================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "entities" not in st.session_state:
    st.session_state.entities = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Guest"
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

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

# ================== Authentication ==================
def login_page():
    st.title("🔐 Welcome to INDIBOT")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                try:
                    with open("users.json", "r") as f:
                        users = json.load(f)
                    user_found = any(u["username"] == username and u["password"] == password for u in users)
                    if user_found:
                        st.session_state.user_logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except FileNotFoundError:
                    st.error("No registered users found. Please register first.")
    with tab2:
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_email = st.text_input("Email ID", key="reg_email")
        if st.button("Register", key="register_button"):
            if not reg_username or not reg_password or not reg_email:
                st.error("All fields are required.")
            else:
                if not os.path.exists("users.json"):
                    with open("users.json", "w") as f:
                        json.dump([], f)
                with open("users.json", "r") as f:
                    users = json.load(f)
                if any(u["username"] == reg_username for u in users):
                    st.error("Username already exists.")
                elif any(u["email"] == reg_email for u in users):
                    st.error("Email already registered.")
                else:
                    users.append({"username": reg_username, "password": reg_password, "email": reg_email})
                    with open("users.json", "w") as f:
                        json.dump(users, f, indent=4)
                    st.success("Registration successful!")

# ================== LLM & Chains ==================
text_generation_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512, temperature=0.7)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
google_search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY) if SERPER_API_KEY else None

# ================== Rule-based KB ==================
RULE_BASED_RESPONSES = {
    "what is your name": "I am INDIBOT, your personal AI assistant!",
    "who created you": "I was created by Shivam Singh Bhadoriya.",
    "how are you": "I'm doing great and always ready to help you!",
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! Ready to chat."
}

def match_intent(query):
    q = query.lower()
    # direct keyword match
    for key, value in RULE_BASED_RESPONSES.items():
        if key in q:
            return value
    # fuzzy match for near queries
    match = get_close_matches(q, RULE_BASED_RESPONSES.keys(), n=1, cutoff=0.75)
    if match:
        return RULE_BASED_RESPONSES[match[0]]
    return None

def analyze_entities_and_sentiment(user_input):
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = TextBlob(user_input).sentiment.polarity
    tone = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
    st.session_state.entities.extend(entities)
    return entities, tone

# ================== Main Chat ==================
def main_app():
    st.title("🧠 INDIBOT AI (Hybrid Search + NLP)")
    st.caption("Understands context, intent, tone, and has memory!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("🎤 Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # NLP Processing
        entities, tone = analyze_entities_and_sentiment(user_question)
        st.info(f"Detected tone: **{tone}** | Entities: {entities if entities else 'None'}")

        with st.spinner("🔄 Thinking..."):
            # Step 1: Rule-based intent
            rule_answer = match_intent(user_question)
            if rule_answer:
                final_response = rule_answer
            else:
                # Step 2: RAG local search
                rag_answer = qa_chain.invoke({"question": user_question, "chat_history": st.session_state.chat_history})
                final_response = rag_answer["answer"]

                # Step 3: Wikipedia fallback
                if not final_response or "i don't know" in final_response.lower():
                    wiki_summary = wikipedia.run(user_question)
                    if wiki_summary and "may refer to" not in wiki_summary.lower():
                        prompt = f"Question: {user_question}\nWikipedia: {wiki_summary}\nAnswer:"
                        final_response = llm.invoke(prompt)
                    elif google_search:
                        final_response = google_search.run(user_question)
                    else:
                        final_response = "I'm sorry, I couldn't find relevant information."

        st.session_state.messages.append({"role": "assistant", "content": final_response})
        with st.chat_message("assistant"):
            st.markdown(final_response)

        # Save chat
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = str(datetime.now().timestamp())
        save_chat_history(st.session_state.current_chat_id, user_question, final_response)

# ================== Sidebar ==================
st.set_page_config(page_title="INDIBOT AI", layout="wide")
with st.sidebar:
    try:
        logo = Image.open("INDIBOT.png")
        st.image(logo, width=120)
    except:
        st.write("🤖 INDIBOT")
    st.header("Settings")
    if st.button("Toggle Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")
    st.header("Profile")
    st.write(f"**User:** {st.session_state.username}")
    if st.button("Logout"):
        st.session_state.user_logged_in = False
        st.session_state.username = "Guest"
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.header("Recent Chats")
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            all_chats = json.load(f)
        for chat in all_chats:
            if st.button(f"📄 {chat['title']}", key=f"chat_{chat['id']}"):
                load_chat(chat["id"])
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.rerun()

# ================== Dark Mode CSS ==================
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #121212; color: #f0f0f0; }
        .stTextInput, .stTextArea { background-color: #333 !important; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #ffffff; color: #000000; }
        </style>
    """, unsafe_allow_html=True)

# ================== Entry Point ==================
if not st.session_state.user_logged_in:
    login_page()
else:
    main_app()
