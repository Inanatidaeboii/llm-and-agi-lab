import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

PDF_FILE_PATH = "../67008_6.pdf"
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "resume_collection"

def ingest_data():
    print(f"Loading PDF : {PDF_FILE_PATH}")
    if not os.path.exists(PDF_FILE_PATH):
        raise ValueError(f"PDF file not found at {PDF_FILE_PATH}")
    
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")
    
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    test_embed = embedding_model.embed_query("test")
    vector_size = len(test_embed)
    print(f"Vector size: {vector_size}")

    client = QdrantClient(url=QDRANT_URL)
    
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}' with size {vector_size}...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

    print("Upserting Vectors to Qdrant...")
    QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=False
    )

    print(f"\nSuccess! Data ingested to Qdrant at '{QDRANT_URL}' in collection '{COLLECTION_NAME}'")
    print(f"Check Dashboard at '{QDRANT_URL}/dashboard' for results")

if __name__ == "__main__":
    ingest_data()