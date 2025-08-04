import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "docs/"
FAISS_INDEX_PATH = "faiss_index.pkl"

def create_vector_db():
    # Load all .txt files from docs/
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    if not documents:
        print("No documents found in docs/ folder.")
        return

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Generate embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    vector_db = FAISS.from_documents(text_chunks, embedding)

    # Save FAISS index
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vector_db, f)

    print(f"FAISS index created with {len(text_chunks)} chunks and saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    create_vector_db()
