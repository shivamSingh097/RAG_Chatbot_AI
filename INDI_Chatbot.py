<<<<<<< HEAD
# INDI_Chatbot.py
# Final merged app: ChatGPT-like UI + local GGUF (optional) + RAG + search fallbacks + persistence
=======
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
import os
import sqlite3
import threading
import queue
import time
from datetime import datetime

import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gtts import gTTS
import pickle
<<<<<<< HEAD

from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Optional Google search (if installed)
try:
    from googlesearch import search
    GOOGLE_ENABLED = True
except Exception:
    GOOGLE_ENABLED = False

# ========== CONFIG ==========
MODEL_GGUF_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # change if using a different model file
FAISS_INDEX_PATH = "faiss_index.pkl"
SQLITE_DB = "indibot.db"

# ========== STREAMLIT PAGE ==========
st.set_page_config(page_title="INDIBOT AI", layout="wide")
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ========== DATABASE ==========
def init_db():
    conn = sqlite3.connect(SQLITE_DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    email TEXT,
                    created_at TEXT
                )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT,
                    created_at TEXT,
                    last_updated TEXT
                )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    sender TEXT,
                    content TEXT,
                    tone TEXT,
                    timestamp TEXT
                )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    question TEXT,
                    tone TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    return conn
=======
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
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5

_db = init_db()
_db_lock = threading.Lock()

<<<<<<< HEAD
def create_user(username, password, email):
    with _db_lock:
        cur = _db.cursor()
        now = datetime.now().isoformat()
        try:
            cur.execute("INSERT INTO users (username,password,email,created_at) VALUES (?, ?, ?, ?)",
                        (username, password, email, now))
            _db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def check_user(username, password):
    with _db_lock:
        cur = _db.cursor()
        cur.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
        r = cur.fetchone()
        return r[0] if r else None

def save_message_db(user_id, chat_id, sender, content, tone):
    with _db_lock:
        cur = _db.cursor()
        now = datetime.now().isoformat()
        if chat_id is None:
            title = content[:60] + "..."
            cur.execute("INSERT INTO chats (user_id,title,created_at,last_updated) VALUES (?, ?, ?, ?)",
                        (user_id, title, now, now))
            chat_id = cur.lastrowid
        cur.execute("INSERT INTO messages (chat_id, sender, content, tone, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (chat_id, sender, content, tone, now))
        cur.execute("UPDATE chats SET last_updated=? WHERE id=?", (now, chat_id))
        _db.commit()
        return chat_id

def get_user_chats(user_id):
    with _db_lock:
        cur = _db.cursor()
        cur.execute("SELECT id, title, last_updated FROM chats WHERE user_id=? ORDER BY last_updated DESC", (user_id,))
        return cur.fetchall()

def get_chat_messages(chat_id):
    with _db_lock:
        cur = _db.cursor()
        cur.execute("SELECT sender, content, timestamp FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
        return cur.fetchall()

def delete_all_user_chats(user_id):
    with _db_lock:
        cur = _db.cursor()
        cur.execute("DELETE FROM messages WHERE chat_id IN (SELECT id FROM chats WHERE user_id=?)", (user_id,))
        cur.execute("DELETE FROM chats WHERE user_id=?", (user_id,))
        _db.commit()

def save_analytics_db(user_id, question, tone):
    with _db_lock:
        cur = _db.cursor()
        now = datetime.now().isoformat()
        cur.execute("INSERT INTO analytics (user_id, question, tone, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, question, tone, now))
        _db.commit()

# ========== NLP ==========

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

@st.cache_resource
def load_intent_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

intent_model = load_intent_model()

INTENT_PATTERNS = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "creator": ["who created you", "who built you", "your developer"],
    "feeling": ["how are you", "how is it going"],
    "name": ["what is your name", "who are you"],
    "bye": ["bye", "goodbye", "see you"]
}
INTENT_RESPONSES = {
    "greeting": "Hello {name}! How can I help you today?",
    "creator": "I was created by Shivam Singh Bhadoriya.",
    "feeling": "I'm just a program, but I'm here to help!",
    "name": "I'm INDIBOT, your AI assistant.",
    "bye": "Goodbye {name}! Have a wonderful day."
}

def classify_intent(user_input):
    try:
        emb1 = intent_model.encode(user_input, convert_to_tensor=True)
        scores = {}
        for intent, phrases in INTENT_PATTERNS.items():
            emb2 = intent_model.encode(phrases, convert_to_tensor=True)
            cosine_scores = util.cos_sim(emb1, emb2)
            scores[intent] = float(np.max(cosine_scores.cpu().numpy()))
        best = max(scores, key=scores.get)
        return best if scores[best] > 0.55 else None
    except Exception:
        return None

def analyze_entities_and_sentiment(user_input):
    try:
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentiment = TextBlob(user_input).sentiment.polarity
        tone = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        return entities, tone
    except Exception:
        return [], "neutral"

# ========== FAISS ==========
@st.cache_resource
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    with open(FAISS_INDEX_PATH, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            return None

vector_db = load_faiss_index()

# ========== Local LLM ==========
@st.cache_resource
def load_local_llm():
    if not os.path.exists(MODEL_GGUF_FILENAME):
        st.warning(f"Local model file not found: {MODEL_GGUF_FILENAME}. LLM responses will be disabled until file is present.")
        return None
    try:
        llm = CTransformers(model=MODEL_GGUF_FILENAME, model_type="mistral", max_new_tokens=512, temperature=0.7)
        return llm
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        return None

llm = load_local_llm()
inference_lock = threading.Lock()
inference_queue = queue.Queue(maxsize=200)

def llm_infer(prompt):
    if llm is None:
        raise RuntimeError("Local model not loaded")
    inference_queue.put(object())
    with inference_lock:
        try:
            inference_queue.get_nowait()
        except Exception:
            pass
        resp = llm.invoke(prompt)
        text = resp if isinstance(resp, str) else str(resp)
        return text

# ========== Search helpers ==========
def duckduckgo_search(query):
    try:
        r = requests.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1}, timeout=8)
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        for obj in data.get("RelatedTopics", []):
            if isinstance(obj, dict) and obj.get("Text"):
                return obj.get("Text")
        return ""
    except Exception:
        return ""

def wikipedia_search(query):
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception:
        return ""

def google_search_snippet(query):
    if not GOOGLE_ENABLED:
        return ""
    try:
        for url in search(query, num=1, stop=1, pause=2):
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, timeout=8, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text() for p in paragraphs[:6])
            if len(text.strip()) > 50:
                return text.strip()
    except Exception:
        return ""
    return ""

# ========== Prompt & utils ==========
def wrap_prompt(user_question, context=""):
    sys_prompt = ("You are INDIBOT, a helpful, concise, and honest AI assistant.\n"
                  "Use the provided context to answer. If unknown, say 'I don't know'.\n\n")
    ctx = f"Context:\n{context}\n\n" if context else ""
    return f"{sys_prompt}{ctx}User: {user_question}\nAssistant:"

def is_low_quality(ans):
    if not ans:
        return True
    s = str(ans).strip()
    if s == "":
        return True
    if len(s.split()) < 4:
        return True
    if "i don't know" in s.lower():
        return True
    return False

# ========== RAG chain ==========
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3}) if vector_db else None
qa_chain = None
if retriever and llm is not None:
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    except Exception:
        qa_chain = None

# ========== UI helpers ==========
def render_sidebar(user_id):
    st.sidebar.title("INDIBOT")
    st.sidebar.markdown("**Conversations**")
    chats = get_user_chats(user_id)
    # New chat button
    if st.sidebar.button("➕ New chat"):
        return None, True
    selected = None
    for cid, title, last in chats:
        if st.sidebar.button(title, key=f"chat_{cid}"):
            selected = cid
            break
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear all chats"):
        delete_all_user_chats(user_id)
        st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.write("Model loaded: " + ("✅" if llm is not None else "❌"))
    st.sidebar.write("Google fallback: " + ("✅" if GOOGLE_ENABLED else "❌"))
    return selected, False

def stream_response_to_placeholder(placeholder, text):
    out = ""
    for ch in text:
        out += ch
        if len(out) % 12 == 0 or ch == "\n":
            placeholder.markdown(out)
            time.sleep(0.02)
    placeholder.markdown(out)

# ========== Auth pages ==========
=======
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
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
def login_page():
    st.title("🔐 INDIBOT Login / Register")
    col1, col2 = st.columns([1, 2])
    with col2:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_btn = st.button("Login")
        if login_btn:
            uid = check_user(username, password)
            if uid:
                st.session_state.user_logged_in = True
                st.session_state.user_id = uid
                st.session_state.username = username
                st.experimental_rerun()
            else:
<<<<<<< HEAD
                st.error("Invalid username/password")
        st.write("---")
        st.write("Register new user")
        new_u = st.text_input("New username", key="reg_username")
        new_p = st.text_input("New password", type="password", key="reg_password")
        new_e = st.text_input("Email", key="reg_email")
        if st.button("Register"):
            if not new_u or not new_p or not new_e:
                st.error("All fields required")
=======
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
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
            else:
                ok = create_user(new_u, new_p, new_e)
                if ok:
                    st.success("Registered! Please login.")
                else:
<<<<<<< HEAD
                    st.error("Username exists.")

# ========== Main chat UI ==========
def main_chat():
    st.sidebar.title(f"Hello, {st.session_state.username} 👋")
    selected_chat_id, new_chat_pressed = render_sidebar(st.session_state.user_id)

    # Main area
    st.markdown("<h2 style='display:flex;align-items:center'>🧠 <span style='margin-left:10px'>INDIBOT AI</span></h2>", unsafe_allow_html=True)
    if new_chat_pressed:
        selected_chat_id = None

    # Display conversation messages
    if "current_chat_messages" not in st.session_state:
        st.session_state.current_chat_messages = []

    # If a chat was selected, load messages from DB
    if selected_chat_id is not None:
        msgs = get_chat_messages(selected_chat_id)
        st.session_state.current_chat_messages = [{"role": "user" if s=="user" else "assistant", "text": c} for s, c, t in msgs]

    # Render messages area
    chat_container = st.container()
    with chat_container:
        for m in st.session_state.current_chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["text"])

    # Input area at bottom
    user_input = st.chat_input("Ask your question...")
    if user_input:
        # append user message immediately to session and DB
        st.session_state.current_chat_messages.append({"role": "user", "text": user_input})
        # show the new message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # persist user message (create chat if needed)
        chat_id = save_message_db(st.session_state.user_id, selected_chat_id, "user", user_input, "neutral")
        # update selected_chat_id so further messages are associated with this chat
        selected_chat_id = chat_id

        # show assistant placeholder while processing
        placeholder = st.empty()
        with st.chat_message("assistant"):
            thinking = placeholder.markdown("*Thinking...*")

        # run pipeline (RAG -> wiki -> ddg -> google -> LLM)
        final_resp = ""
        wiki_summary = ""
        ddg_result = ""
        google_text = ""
        tone = "neutral"
        try:
            # analyze tone & entities (non-blocking tiny cost)
            _, tone = analyze_entities_and_sentiment(user_input)
            intent = classify_intent(user_input)
            if intent and intent in INTENT_RESPONSES:
                final_resp = INTENT_RESPONSES[intent].format(name=st.session_state.username)
            else:
                # 1) RAG
                if qa_chain:
                    try:
                        rag_answer = qa_chain.invoke({"question": user_input, "chat_history": memory.buffer})
                        if isinstance(rag_answer, dict):
                            final_resp = rag_answer.get("answer") or rag_answer.get("result") or ""
                        else:
                            final_resp = str(rag_answer)
                    except Exception:
                        final_resp = ""
                # 2) Wikipedia
                if is_low_quality(final_resp):
                    wiki_summary = wikipedia_search(user_input)
                    if wiki_summary:
                        final_resp = wiki_summary
                # 3) DuckDuckGo
                if is_low_quality(final_resp):
                    ddg_result = duckduckgo_search(user_input)
                    if ddg_result:
                        final_resp = ddg_result
                # 4) Google
                if GOOGLE_ENABLED and is_low_quality(final_resp):
                    google_text = google_search_snippet(user_input)
                    if google_text:
                        final_resp = google_text
                # 5) LLM synthesis
                if is_low_quality(final_resp):
                    ctx = ""
                    for c in (wiki_summary, ddg_result, google_text):
                        if c:
                            ctx += c + "\n\n"
                    prompt = wrap_prompt(user_input, ctx)
                    if llm is None:
                        final_resp = "Local model not available. Please download the GGUF model file to enable full LLM responses."
                    else:
                        # Blocking inference, but stream characters to placeholder for UX
                        try:
                            text = llm_infer(prompt)
                            stream_response_to_placeholder(placeholder, text)
                            final_resp = text
                        except Exception as e:
                            final_resp = f"Error generating response: {e}"
            # update DB with assistant response
            save_message_db(st.session_state.user_id, selected_chat_id, "assistant", final_resp, tone)
            save_analytics_db(st.session_state.user_id, user_input, tone)
        except Exception as e_outer:
            final_resp = f"Unexpected error: {e_outer}"
            save_message_db(st.session_state.user_id, selected_chat_id, "assistant", final_resp, tone)

        # replace placeholder if still present
        try:
            placeholder.markdown(final_resp)
        except Exception:
            pass

        # show TTS option and final message (already displayed via placeholder)
        with st.chat_message("assistant"):
            st.markdown(final_resp)
            if st.button("🔊 Speak", key=f"tts_{int(time.time())}"):
                try:
                    tfile = f"resp_{int(time.time())}.mp3"
                    tts = gTTS(final_resp)
                    tts.save(tfile)
                    st.audio(tfile)
                    os.remove(tfile)
                except Exception as e:
                    st.error(f"TTS failed: {e}")

        # update conversation list in sidebar by reloading
        st.experimental_rerun()

# ========== Entrypoint ==========
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
=======
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
    st.warning("⚠️ Serper API key not found. Google Search functionality will be disabled.")


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
                except Exception as e:
                    st.error(f"RAG query failed: {e}")
                    final_response = llm.invoke(f"Answer the following question: {user_question}")

                if not final_response or "i don't know" in final_response.lower():
                    wiki_summary = wikipedia.run(user_question)
                    if wiki_summary and not "no good match found" in wiki_summary.lower():
                        prompt = f"Question: {user_question}\nWikipedia: {wiki_summary}\nAnswer:"
                        final_response = llm.invoke(prompt)
                    elif GoogleSearch:
                        final_response = GoogleSearch.run(user_question)
                    else:
                        final_response = "I couldn't find relevant information using my internal knowledge, Wikipedia, or Google Search."
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        with st.chat_message("assistant"):
            st.markdown(final_response)
            if st.button("🔊 Speak", key=f"voice_{len(st.session_state.messages)}"):
                audio_file = speak_text(final_response)
                st.audio(audio_file)
                os.remove(audio_file)
        
        save_chat_history(str(datetime.now().timestamp()), user_question, final_response, tone)
        st.session_state.analytics.append({"question": user_question, "tone": tone, "timestamp": datetime.now().isoformat()})

# ===================== Analytics Tab =====================
def analytics_tab():
    st.title("📊 Chat Analytics")
    if not st.session_state.analytics:
        st.write("No analytics yet.")
        return
    
    df = [{"Tone": a["tone"], "Time": a["timestamp"]} for a in st.session_state.analytics]
    
    chart = alt.Chart(alt.Data(values=df)).mark_bar().encode(
        x=alt.X("Tone:N", axis=None), 
        y=alt.Y("count()", title="Number of Queries")
    )
    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("Recent Queries")
    for i, entry in enumerate(reversed(st.session_state.analytics)):
        st.write(f"{i+1}. **Query:** {entry['question']} | **Tone:** {entry['tone']} | **Time:** {entry['timestamp'][:19]}")

# ===================== News Tab =====================
def news_tab():
    st.title("📰 Top News Headlines")
    news = fetch_news()
    for n in news: 
        st.write(n)

# ===================== Main Application Entry Point =====================
st.set_page_config(page_title="INDIBOT AI", layout="wide")
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5

if not st.session_state.user_logged_in:
    login_page()
else:
<<<<<<< HEAD
    main_chat()
=======
    # Sidebar for navigation
    st.sidebar.title(f"Hello, {st.session_state.user_name} 👋")
    tabs = st.sidebar.radio("Choose a page", ["Chat", "Analytics", "News"])
    if st.sidebar.button("Toggle Dark Mode"): 
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    if st.sidebar.button("Logout"):
        st.session_state.user_logged_in = False
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # Apply dark mode style
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

    # Render the selected page
    if tabs == "Chat":
        main_chat()
    elif tabs == "Analytics":
        analytics_tab()
    elif tabs == "News":
        news_tab()
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
