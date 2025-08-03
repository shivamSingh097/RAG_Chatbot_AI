import os
import langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

Data_Path = "docs/"
DB_Path = "chroma_db/"

def create_vector_db():
    loader = DirectoryLoader(Data_Path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Creating vector database with {len(text)} chunks...")
    vector_db = Chroma.from_documents(text, embedding, persist_directory=DB_Path)
    vector_db.persist()
    print("Vector database created and persisted.")

if __name__ == "__main__":
    create_vector_db()