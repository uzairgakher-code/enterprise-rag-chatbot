from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Make sure your loaders exist
from loaders.file_loader import load_files
from loaders.web_loader import load_urls

import config
import os

print("🔄 Loading data...")

# Load documents from files
docs = load_files(config.DATA_FILES_PATH)

# Load documents from URLs
docs += load_urls(config.URLS_FILE_PATH)

print(f"📄 Loaded {len(docs)} documents")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

chunks = splitter.split_documents(docs)
print(f"📝 Split into {len(chunks)} chunks")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL
)

# Create FAISS vectorstore from chunks
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save locally
os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
vectorstore.save_local(config.VECTOR_DB_PATH)

print("✅ Knowledge base created successfully")
