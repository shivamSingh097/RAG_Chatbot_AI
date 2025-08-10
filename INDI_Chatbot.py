import os
import pickle
import sys
import json
import streamlit as st
from PIL import Image
from huggingface_hub import login
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import Tool
import requests
from datetime import datetime
import altair as alt
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gtts import gTTS

# ===================== API Keys =====================
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", st.secrets.get("HUGGINGFACEHUB_API_TOKEN"))
if not HF_TOKEN:
    st.error("❌ Hugging Face token is missing. Add it to .streamlit/secrets.toml")
    st.stop()

try:
    login(HF_TOKEN)
except Exception:
    st.warning("⚠️ Hugging Face login failed. Check token in Streamlit secrets.")

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", st.secrets.get("SERPER_API_KEY"))
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")

# ===================== Check FAISS Index =====================
FAISS_INDEX_PATH = "faiss_index.pkl"
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("❌ FAISS index not found. Run ingest.py first to create it.")
    st.stop()

with open(FAISS_INDEX_PATH, "rb") as f:
    vector_db = pickle.load(f)

# ===================== SpaCy Model and Sentence Transformer =====================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.info("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

@st.cache_resource
def load_intent_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

intent_model = load_intent_model()

# ---------------- Intents ----------------
INTENT_RESPONSES = {
    "greeting": "Hello {name}! How can I help you today?",
    "creator": "I was created by Shivam Singh Bhadoriya.",
    "feeling": "I'm just a bunch of code, but feeling great! 😎",
    "name": "I'm INDIBOT, your AI assistant.",
    "bye": "Goodbye {name}! Have a wonderful day."
}
INTENT_PATTERNS = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "creator": ["who created you", "who built you", "your developer"],
    "feeling": ["how are you", "how is it going"],
    "name": ["what is your name", "who are you"],
    "bye": ["bye", "goodbye", "see you"]
}

def classify_intent(user_input):
    scores = {}
    for intent, phrases in INTENT_PATTERNS.items():
        embeddings1 = intent_model.encode(user_input, convert_to_tensor=True)
        embeddings2 = intent_model.encode(phrases, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        max_score = float(np.max(cosine_scores.cpu().numpy()))
        scores[intent] = max_score
    best_intent = max(scores, key=scores.get)
    return best_intent if scores[best_intent] > 0.55 else None

def analyze_entities_and_sentiment(user_input):
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = TextBlob(user_input).sentiment.polarity
    tone = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
    return entities, tone

# ===================== Session State =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Guest"
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "analytics" not in st.session_state:
    st.session_state.analytics = []
if "loading" not in st.session_state:
    st.session_state.loading = False


# ---------------- Chat_History ----------------
USERS_FILE = "users.json"
CHAT_HISTORY_FILE = "chat_history.json"

def save_chat_history(chat_id, user_message, ai_message, tone):
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            history = json.load(f)
    chat_found = False
    for chat in history:
        if chat["id"] == chat_id:
            chat["conversation"].append({"user": user_message, "ai": ai_message, "tone": tone})
            chat["last_updated"] = datetime.now().isoformat()
            chat_found = True
            break
    if not chat_found:
        history.append({
            "id": chat_id,
            "title": user_message[:30] + "...",
            "last_updated": datetime.now().isoformat(),
            "conversation": [{"user": user_message, "ai": ai_message, "tone": tone}]
        })
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

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
                        st.session_state.user_name = username # Set user_name after login
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

# ===================== LLM (Hugging Face Hub) =====================
@st.cache_resource
def load_llm():
    text_generation_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

llm = load_llm()

# ===================== Retrieval QA (with conversation memory) =====================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
GoogleSearch = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY) if SERPER_API_KEY else None
if not SERPER_API_KEY:
    st.warning("⚠️ Serper API key not found. GoogleSearch functionality will be disabled.")


# ===================== Helpers =====================
def fetch_news():
    if not NEWS_API_KEY:
        return ["⚠️ NEWS_API_KEY not configured. News functionality is disabled."]
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    try:
        r = requests.get(url)
        r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        articles = r.json().get("articles", [])
        return [f"- {a['title']}" for a in articles[:5]]
    except requests.exceptions.RequestException as e:
        return [f"Failed to fetch news: {e}"]

def speak_text(text):
    tts = gTTS(text)
    file_path = "response.mp3"
    tts.save(file_path)
    return file_path

# ===================== Main Chat Logic =====================
def main_chat():
    st.title("🧠 INDIBOT AI - Chat")
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_question := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"): 
            st.markdown(user_question)
        
        entities, tone = analyze_entities_and_sentiment(user_question)
        intent = classify_intent(user_question)
        st.info(f"Tone: **{tone}**, Entities: {entities if entities else 'None'}")
        
        with st.spinner("Thinking..."):
            final_response = ""
            if intent and intent in INTENT_RESPONSES:
                name = st.session_state.user_name or "Friend"
                final_response = INTENT_RESPONSES[intent].format(name=name)
            else:
                try:
                    rag_answer = qa_chain.invoke({"question": user_question, "chat_history": st.session_state.chat_history})
                    final_response = rag_answer["answer"]
                except
