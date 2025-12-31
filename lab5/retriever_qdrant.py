import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "resume_collection"

def test_retrieval():
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    
    print("\n" + "=" * 50)
    query = input("Enter query: ")

    if not query.strip():
        return
    
    print(f"\n --- Searching for : '{query}' ---")

    results = vector_store.similarity_search_with_score(query, k=3)

    for doc, score in results:
        print(f"\n[Score: {score:.4f}] Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
        print(f"Content snippet: {doc.page_content[:150]}...")

    print("\n" + "=" * 50)
    print("--- Testing Metadata Filter (Page = 0) ---")

    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="page",
                match=models.MatchValue(value=0)
            )
        ]
    )

    filtered_results = vector_store.similarity_search(
        query,
        k=3,
        filter=filter_condition
    )

    if filtered_results:
        for doc in filtered_results:
            print(f"\n[Filtered] Page: {doc.metadata.get('page')}")
            print(f"Content: {doc.page_content[:150]}...")
    else:
        print("No results found")

if __name__ == '__main__':
    test_retrieval()