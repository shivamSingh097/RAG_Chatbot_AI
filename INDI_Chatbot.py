# INDI_Chatbot.py
# Final merged app: ChatGPT-like UI + local GGUF (optional) + RAG + search fallbacks + persistence
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
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gtts import gTTS
import pickle

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
from ctransformers import AutoModelForCausalLM as CTransformers
from langchain_community.llms import HuggingFaceHub

# Path to local GGUF model file
MODEL_GGUF_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

@st.cache_resource
def load_local_llm():
    if not os.path.exists(MODEL_GGUF_FILENAME):
        st.error(f"Local model file not found: {MODEL_GGUF_FILENAME}. LLM responses will be disabled until file is present.")
        return None

    return CTransformers(
        model=MODEL_GGUF_FILENAME,
        model_type="mistral",
        config={"temperature": 0.7, "max_new_tokens": 512}
    )

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

_db = init_db()
_db_lock = threading.Lock()

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

import wikipedia

def fetch_wikipedia_summary(query):
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return "I couldn't find information about that."

        # Try to pick the most relevant result
        for title in search_results:
            if query.lower() in title.lower():
                page = wikipedia.page(title)
                return page.summary

        # Fallback: just take the first result
        page = wikipedia.page(search_results[0])
        return page.summary

    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"


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
        st.rerun()
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
                st.rerun()
            else:
                st.error("Invalid username/password")
        st.write("---")
        st.write("Register new user")
        new_u = st.text_input("New username", key="reg_username")
        new_p = st.text_input("New password", type="password", key="reg_password")
        new_e = st.text_input("Email", key="reg_email")
        if st.button("Register"):
            if not new_u or not new_p or not new_e:
                st.error("All fields required")
            else:
                ok = create_user(new_u, new_p, new_e)
                if ok:
                    st.success("Registered! Please login.")
                else:
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
        st.rerun()

# ========== Entrypoint ==========
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

if not st.session_state.user_logged_in:
    login_page()
else:
    main_chat()
