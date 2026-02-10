from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K
import os

def get_retriever():
    # Create embeddings object
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Ensure the vectorstore path exists
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"Vectorstore not found at {VECTOR_DB_PATH}. Please run ingest.py first.")

    # Load FAISS vectorstore
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings
    )

    # Return a retriever with top-k search
    return vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )
