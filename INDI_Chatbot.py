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
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime
import pysqlite3
import sys

# ===================== Fix for ChromaDB and sqlite3 on Streamlit Cloud =====================
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
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

# ===================== LLM & Chains =====================
# This part is outside the login function, so it only runs after login
text_generation_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Use a standard retrieval chain instead of an agent to avoid the AttributeError
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ===================== Main App Logic =====================
def get_intent_response(user_msg):
    """Check rule-based knowledge base for a quick response"""
    knowledge_base = {
        "what is your name": "🤖 I'm IndiBot, your friendly AI assistant!",
        "how are you": "😊 I'm just code, but thanks for asking! How can I help?",
        "tell me a joke": "😄 Why did the developer go broke? Because they used up all their cache!",
        "what can you do": "🧠 I can answer questions, search the web, and more. Try me!",
    }

    cleaned = user_msg.lower().strip()
    return knowledge_base.get(cleaned, None)


def extract_intents_keywords(text):
    """Simple NLP keyword-based intent recognition"""
    keywords = {
        "name": "🤖 I'm IndiBot, your AI assistant!",
        "hello": "👋 Hello! How can I help you today?",
        "hi": "👋 Hi there! Ask me anything.",
        "order pizza": "🍕 Sure! Please provide size and toppings (Demo action triggered)",
        "thank you": f"You're welcome, {st.session_state.get('user_name', 'friend')}! 😊",
    }

    for keyword in keywords:
        if keyword in text.lower():
            return keywords[keyword]
    return None


def main_app():
    st.title("🧠 IndiBot AI")
    st.write("Ask me anything about AI, Python, Economy, General Knowledge or Live Web Search! ✨")

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("🎤 Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("🔄 Thinking..."):
            try:
                # Step 1: Rule-based (Exact Match)
                rule_response = get_intent_response(user_question)
                if rule_response:
                    final_response = rule_response

                else:
                    # Step 2: NLP-style Intent Matching
                    intent_response = extract_intents_keywords(user_question)
                    if intent_response:
                        final_response = intent_response

                    else:
                        # Step 3: Local Knowledge Base (RAG)
                        local_answer = qa_chain.invoke({"question": user_question, "chat_history": st.session_state.chat_history})
                        final_response = local_answer['answer']

                        if "i don't know" in final_response.lower() or "could not find" in final_response.lower() or "unanswerable" in final_response.lower():
                            st.info("🤔 Couldn't find locally. Trying Wikipedia...")

                            # Step 4: Wikipedia fallback
                            wiki_summary = wikipedia.run(user_question)
                            if "unanswerable" in wiki_summary.lower() or len(wiki_summary.strip()) < 10:
                                st.info("🌐 Wikipedia failed. Trying Google Search (Parser API)...")

                                # Step 5: Parser API Search fallback
                                try:
                                    parser_result = google_parser_search.run(user_question)
                                    final_response = f"🌍 Google says: {parser_result}"
                                except Exception as e:
                                    final_response = "❌ All search methods failed. Please rephrase."

                            else:
                                # Final answer synthesis from wiki
                                prompt = f"""
                                You are a helpful assistant. Based on the following Wikipedia result, answer the user's question.
                                Question: {user_question}
                                Wiki Summary: {wiki_summary}
                                Answer:"""
                                final_response = llm.invoke(prompt)

                # Show bot reply
                with st.chat_message("assistant"):
                    st.markdown(final_response)

                # Store message
                st.session_state.messages.append({"role": "assistant", "content": final_response})

                # Save chat to file
                if st.session_state.current_chat_id is None:
                    st.session_state.current_chat_id = str(datetime.now().timestamp())
                save_chat_history(st.session_state.current_chat_id, user_question, final_response)

            except Exception as e:
                st.error("❌ An error occurred.")
                st.exception(e)


# ===================== Sidebar and Entry Point =====================
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
        
if not st.session_state.user_logged_in:
    login_page()
else:
    main_app()
