import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

# Web Search
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool

# Transformers
from transformers import pipeline
import re

# Load environment variables
load_dotenv()

# --- Set up embedding model and vector store ---
DB_PATH = "chroma_db/"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

# --- Set up LLM ---
text_generation_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,
    max_new_tokens=100,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- Custom Prompt for short answer ---
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant. Use the context below to answer the question in one or two precise sentences.
Avoid extra explanation or unrelated facts.

Context:
{context}

Question: {question}
Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# --- Google Search Tool ---
use_web_search = True
if use_web_search:
    search = GoogleSerperAPIWrapper()
    web_search_tool = Tool(
        name="Google Search",
        description="A tool to search the web for information.",
        func=search.run
    )

# --- Streamlit UI Config ---
st.set_page_config(page_title="INDI_Chatbot", page_icon="🤖", layout="centered")

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background-color: #0f1116;
    color: #ffffff;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1c1e26, #2a2d3e);
}
h1, h2, h3 {
    color: #00c8ff;
}
input, textarea {
    background-color: #1a1a1a !important;
    color: white !important;
}
.stTextInput>div>div>input {
    border: 2px solid #00c8ff;
    border-radius: 12px;
    padding: 0.75rem;
}
.stButton>button {
    background-color: #00c8ff;
    color: black;
    font-weight: bold;
    border-radius: 10px;
}
.message-bubble {
    padding: 1rem;
    border-radius: 12px;
    background-color: #1e1f25;
    margin: 10px 0;
    color: white;
    font-size: 1.1rem;
}
.success-bubble {
    border-left: 5px solid #00c8ff;
}
.info-bubble {
    border-left: 5px solid #00ffa1;
}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🧠 INDI_Chatbot (Vocal for Local)")
st.markdown("Ask any question. Get quick, point-to-point answers from AI & Web.")

# --- Input ---
user_question = st.text_input("❓ Ask your question here:", placeholder="e.g. When was Steve Jobs born?")

if user_question:
    # --- Local Answer ---
    st.markdown("#### 🤖 From Local Knowledge Base")
    try:
        local_answer = qa_chain.run(user_question).strip()
        if local_answer:
            st.markdown(f"<div class='message-bubble success-bubble'>{local_answer}</div>", unsafe_allow_html=True)
        else:
            st.warning("No relevant answer found in local knowledge base.")
    except Exception as e:
        st.error(f"Local QA failed: {e}")

    # --- Web Answer ---
    if use_web_search:
        st.markdown("#### 🌐 From Web Search")
        try:
            web_result = web_search_tool.run(user_question)
            sentences = re.split(r'(?<=[.!?])\s+', web_result.strip())
            short_answer = sentences[0] if sentences else web_result
            st.markdown(f"<div class='message-bubble info-bubble'>{short_answer}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning("Web search failed. Check SERPER_API_KEY or internet.")
            st.error(str(e))
