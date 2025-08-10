<<<<<<< HEAD
# INDI_Chatbot.py (merged final)
=======
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
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
<<<<<<< HEAD
# Removed langchain_community DuckDuckGo dependency to avoid extra install
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import altair as alt
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gtts import gTTS
from dotenv import load_dotenv
import wikipedia  # pip install wikipedia

# ===================== Streamlit config =====================
st.set_page_config(page_title="INDIBOT AI", layout="wide")
load_dotenv()

# ===================== API Keys =====================
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY") or os.getenv("NEWS_API_KEY")

if not HF_TOKEN:
    st.error("❌ Hugging Face token is missing.")
    st.stop()
if not SERPER_API_KEY:
    st.warning("⚠️ Serper API key not found. Google Search functionality will be disabled.")
if not NEWS_API_KEY:
    st.warning("⚠️ News API key not found. News tab functionality will be disabled.")

# ===================== Check FAISS Index =====================
FAISS_INDEX_PATH = "faiss_index.pkl"
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("❌ FAISS index not found. Run ingest.py first to create it.")
    st.stop()

with open(FAISS_INDEX_PATH, "rb") as f:
    vector_db = pickle.load(f)

# ===================== SpaCy & Sentence Transformer =====================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
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
        scores[intent] = float(np.max(cosine_scores.cpu().numpy()))
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
    st.session_state.chat_history = []  # tuples like ("user"/"assistant"/"system", text)
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
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                try:
                    with open(USERS_FILE, 'r') as f:
                        users = json.load(f)
                    if any(u['username'] == username and u['password'] == password for u in users):
                        st.session_state.user_logged_in = True
                        st.session_state.username = username
                        st.session_state.user_name = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except FileNotFoundError:
                    st.error("No registered users found.")

    with tab2:
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_email = st.text_input("Email ID", key="reg_email")
        if st.button("Register"):
            if not reg_username or not reg_password or not reg_email:
                st.error("All fields required.")
            else:
                if not os.path.exists(USERS_FILE):
                    with open(USERS_FILE, "w") as f:
                        json.dump([], f)
                with open(USERS_FILE, 'r') as f:
                    users = json.load(f)
                if any(u['username'] == reg_username for u in users):
                    st.error("Username already exists.")
                elif any(u['email'] == reg_email for u in users):
                    st.error("Email already registered.")
                else:
                    users.append({"username": reg_username, "password": reg_password, "email": reg_email})
                    with open(USERS_FILE, 'w') as f:
                        json.dump(users, f, indent=4)
                    st.success("Registration successful!")

# ===================== LLM =====================
@st.cache_resource
def load_llm():
    text_generation_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.7
=======
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool

# ===================== API Keys =====================
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

# ===================== LLM (Hugging Face Hub) =====================
text_generation_pipeline = pipeline(
    "text2text-generation",  # For FLAN-T5
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# ===================== Retrieval QA (with conversation memory) =====================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# ===================== Google Search (Serper API) =====================
use_web_search = bool(SERPER_API_KEY)
if use_web_search:
    search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    web_search_tool = Tool(
        name="Google Search",
        description="Fetch real-time web search results",
        func=search.run
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

<<<<<<< HEAD
llm = load_llm()

# ===================== Retrieval (RAG) =====================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# ===================== Helpers: news, speak =====================
def fetch_news():
    if not NEWS_API_KEY:
        return ["⚠️ NEWS_API_KEY not configured."]
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            f"📰 **{a['title']}**\n\n{a.get('description', '')}\n\nRead more: {a.get('url', '')}"
            for a in articles[:5]
        ]
    except requests.exceptions.RequestException as e:
        return [f"Failed to fetch news: {e}"]

def speak_text(text):
    tts = gTTS(text)
    file_path = "response.mp3"
    tts.save(file_path)
    return file_path

# ===================== Simple DuckDuckGo HTML search fallback =====================
def duckduckgo_search(query):
    try:
        url = "https://duckduckgo.com/html/"
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.post(url, data=params, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        # Try to get snippet text if any result
        results = soup.select("div.result__snippet")
        if results:
            return results[0].get_text(strip=True)
        # fallback to link titles
        links = soup.find_all("a", class_="result__a")
        if links:
            return links[0].get_text(strip=True)
        return ""
    except Exception as e:
        return ""

# ===================== Wikipedia summary helper =====================
def wikipedia_search(query):
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception:
        return ""

# ===================== Special-case scraper: States & Union Territories of India =====================
def get_states_and_uts():
    try:
        url = "https://en.wikipedia.org/wiki/States_and_union_territories_of_India"
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        # The page has a table listing states — we'll extract unique names from the page's "States and union territories" section
        items = []
        # First try to find the lists under the content that list states/UTs (some pages use ul lists)
        # Fallback: parse the big table rows
        # Look for table with 'wikitable' and headers containing 'State' or 'Union territory'
        tables = soup.find_all("table", {"class": "wikitable"})
        for table in tables:
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if cols:
                    # Many tables put the name in first or second column; try both
                    name = cols[0].get_text(strip=True)
                    # remove footnote numbers
                    name = " ".join(name.split())
                    if name and name not in items:
                        items.append(name)
        # If this didn't find enough (sometimes page format different), try extracting from bullet lists
        if len(items) < 25:
            for ul in soup.select("div.mw-parser-output > ul"):
                for li in ul.find_all("li", recursive=False):
                    text = li.get_text(strip=True)
                    if text and len(text) < 60 and "," not in text:  # heuristic to avoid long descriptions
                        if text not in items:
                            items.append(text)
                    if len(items) >= 36:
                        break
                if len(items) >= 36:
                    break
        # Final clean: dedupe and return
        cleaned = []
        for it in items:
            # remove bracketed refs like [1]
            cleaned_name = BeautifulSoup(it, "html.parser").get_text()
            cleaned_name = "".join(ch for ch in cleaned_name if ch != "\xa0")
            cleaned.append(cleaned_name)
        cleaned_unique = []
        for c in cleaned:
            if c not in cleaned_unique:
                cleaned_unique.append(c)
        # Often list length will be 36 (28 states + 8 UTs) or similar
        return cleaned_unique
    except Exception:
        return []

# ===================== Follow-up resolver =====================
def resolve_follow_up(user_question):
    """Rewrite short/follow-up questions into standalone using last assistant/system context."""
    # If user writes a long standalone question, return as-is
    if not st.session_state.chat_history:
        return user_question

    # Heuristics for follow-up: short (<6 words) or contains pronouns referring to previous subject
    tokens = user_question.strip().split()
    follow_up_triggers = ['them', 'it', 'those', 'who', 'what', 'where', 'which', 'when', 'he', 'she', 'they', 'name', 'name them', 'can you name']
    short_follow_up = len(tokens) < 6 or any(tok.lower() in user_question.lower() for tok in follow_up_triggers)

    if not short_follow_up:
        return user_question

    # Get last assistant or system message text to use as context
    last_context = ""
    for role, msg in reversed(st.session_state.chat_history):
        if role in ("assistant", "system") and msg:
            last_context = msg
            break

    if not last_context:
        return user_question

    # Construct resolved question
    resolved = f"{last_context}\nUser: {user_question}"
    # Keep it short: ask LLM to rewrite into standalone question using the context
    try:
        prompt = f"Given the context:\n{last_context}\n\nRewrite the user's follow-up '{user_question}' as a full standalone question."
        rewritten = llm.invoke(prompt)
        if isinstance(rewritten, str) and len(rewritten.strip()) > 0:
            return rewritten.strip()
    except Exception:
        pass

    # Fallback: naive concatenation
    return f"{last_context} {user_question}"

# ===================== Main Chat (merged & improved) =====================
def main_chat():
    st.title("🧠 INDIBOT AI - Chat")

    # render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if user_question := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            # Analyze intent/entities/tone (on the original user_question for UI display)
            entities, tone = analyze_entities_and_sentiment(user_question)
            intent = classify_intent(user_question)
            st.info(f"Tone: **{tone}**, Entities: {entities if entities else 'None'}")

            with st.spinner("Thinking..."):
                final_response = ""
                wiki_summary = ""
                google_result = ""

                def is_low_quality(answer):
                    if not answer or str(answer).strip() == "":
                        return True
                    if len(str(answer).split()) < 5:
                        return True
                    if "i don't know" in str(answer).lower():
                        return True
                    return False

                # 1️⃣ Intent responses (quick local intents)
                if intent and intent in INTENT_RESPONSES:
                    name = st.session_state.user_name or "Friend"
                    final_response = INTENT_RESPONSES[intent].format(name=name)

                else:
                    # Resolve follow-up into standalone question
                    resolved_question = resolve_follow_up(user_question)

                    # Special-case: ask for states & union territories
                    lowq = is_low_quality("")  # just to reuse function
                    if any(kw in resolved_question.lower() for kw in ["states and union territories", "states & union territories", "states and union territories of india", "name all states", "name all union territories"]) or \
                       ("states" in resolved_question.lower() and "union" in resolved_question.lower()):
                        items = get_states_and_uts()
                        if items:
                            # try to detect if they asked for count + names
                            final_response = f"India has {len(items)} states and union territories combined. Here are the names:\n\n" + ", ".join(items)
                        else:
                            # if scraper fails, fallback to other sources below
                            final_response = ""

                    # 2️⃣ RAG retrieval (only if we still don't have a good answer)
                    if is_low_quality(final_response):
                        try:
                            rag_answer = qa_chain.invoke({"question": resolved_question, "chat_history": st.session_state.chat_history})
                            # LangChain outputs can vary; try common keys
                            if isinstance(rag_answer, dict):
                                final_response = rag_answer.get("answer") or rag_answer.get("output_text") or rag_answer.get("result") or ""
                            else:
                                final_response = str(rag_answer)
                        except Exception as e:
                            st.error(f"RAG query failed: {e}")
                            final_response = ""

                    # 3️⃣ Wikipedia fallback
                    if is_low_quality(final_response):
                        wiki_summary = wikipedia_search(resolved_question)

                    # 4️⃣ DuckDuckGo fallback (only if wiki empty)
                    if is_low_quality(final_response):
                        if wiki_summary and len(wiki_summary.strip()) > 10:
                            # keep wiki_summary
                            pass
                        else:
                            google_result = duckduckgo_search(resolved_question)

                    # 5️⃣ Merge/synthesize with LLM if needed
                    if is_low_quality(final_response):
                        combined_context = ""
                        if wiki_summary and "no good match found" not in wiki_summary.lower():
                            combined_context += f"Wikipedia: {wiki_summary}\n\n"
                        if google_result:
                            combined_context += f"DuckDuckGo: {google_result}\n\n"

                        if combined_context.strip():
                            prompt = f"Question: {resolved_question}\n\nContext:\n{combined_context}\nAnswer concisely:"
                            try:
                                llm_response = llm.invoke(prompt)
                                final_response = llm_response if isinstance(llm_response, str) else str(llm_response)
                            except Exception as e:
                                final_response = "I couldn't find relevant information from Wikipedia or DuckDuckGo."
                        else:
                            # last resort: say cannot find
                            final_response = "I couldn't find relevant information from Wikipedia or DuckDuckGo."

                    # 6️⃣ Store the contexts used so follow-ups can reuse them
                    if wiki_summary:
                        st.session_state.chat_history.append(("system", f"[Wikipedia context] {wiki_summary}"))
                    if google_result:
                        st.session_state.chat_history.append(("system", f"[Search context] {google_result}"))

                # 7️⃣ Present answer
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                with st.chat_message("assistant"):
                    st.markdown(final_response)
                    if st.button("🔊 Speak", key=f"voice_{len(st.session_state.messages)}"):
                        audio_file = speak_text(final_response)
                        st.audio(audio_file)
                        try:
                            os.remove(audio_file)
                        except Exception:
                            pass

                # Save analytics & persistent chat history file
                save_chat_history(str(datetime.now().timestamp()), user_question, final_response, tone)
                st.session_state.analytics.append({"question": user_question, "tone": tone, "timestamp": datetime.now().isoformat()})

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()

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

# ===================== News Tab =====================
def news_tab():
    st.title("📰 Top News Headlines")
    for n in fetch_news():
        st.markdown(n)

# ===================== Main Entry =====================
if not st.session_state.user_logged_in:
    login_page()
else:
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

    if tabs == "Chat":
        main_chat()
    elif tabs == "Analytics":
        analytics_tab()
    elif tabs == "News":
        news_tab()
=======
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
st.title("🧠 INDIBOT")
st.markdown("Ask me anything about AI, Python, Economy, General Knowledge or Live Web Search! ✨")

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

        # ============ Run Web Search if API available ============
        if use_web_search:
            try:
                web_result = web_search_tool.run(user_question)
                st.markdown("### 🌐 Web Search Result")
                st.info(web_result)
            except Exception as e:
                st.warning("⚠️ Web search failed.")
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
>>>>>>> 2f9c9a04f5722a25f010696b2df418cb449123e5
