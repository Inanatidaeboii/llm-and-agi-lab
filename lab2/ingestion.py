import os
import time
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

PDF_FILE_PATH = "67008_6.pdf"
VECTOR_DB_PATH = "./chroma_db_data"
BATCH_SIZE = 10
DELAY = 2

# def process_in_batches(vector_store, splits:List[Document]):
#     total_splits = len(splits)
#     for i in range(0, total_splits, BATCH_SIZE):
#         batch = splits[i:i + BATCH_SIZE]

#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 vector_store.add_documents(batch)
#                 print(f"Processed batch {i//BATCH_SIZE + 1}/{(total_splits + BATCH_SIZE - 1)//BATCH_SIZE}")
#                 break
#             except Exception as e:
#                 print(f"Error on attempt {attempt+1}: {e}")
#                 if attempt < max_retries - 1:
#                     wait_time = (attempt + 1) * 5
#                     print(f"Waiting {wait_time} seconds before retrying...")
#                     time.sleep(wait_time)
#                 else:
#                     print(f"Failed to process this batch.")
#                     raise
#         time.sleep(DELAY)

def ingest_data():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File not found at {PDF_FILE_PATH}")
        return
        
    print(f"Loading PDF: {PDF_FILE_PATH}...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")

    print("Creating Vector Store...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=VECTOR_DB_PATH,
    )

    print(f"Success! Vector Database saved to '{VECTOR_DB_PATH}'")
    
if __name__ == "__main__":
    start_time = time.time()
    ingest_data()
    print(f"Finished in {time.time() - start_time:.2f} seconds.")    