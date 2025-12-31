import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "resume_collection"

@tool
def search_legal_documents(query: str, page_filter: Optional[int] = None) -> str:
    """
    ค้นหาข้อมูลกฎหมาย ระเบียบ หรือข้อบังคับเกี่ยวกับการจัดการสิ่งปฏิกูล
    ใช้เครื่องมือนี้เมื่อผู้ใช้งานถามเกี่ยวกับเนื้อหาในเอกสาร
    
    Args:
        query: คำค้นหาหลัก หรือคำถามของผู้ใช้งาน (Search Keywords)
        page_filter: (Optional) เลขหน้าเอกสารที่ต้องการระบุเจาะจง เช่น 1, 69. ถ้าไม่ระบุให้เป็น None
    """
    try:
        print(f"\n [Tool Call] Searching: '{query}' | Page Filter: {page_filter}")

        client = QdrantClient(url=QDRANT_URL)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model,
        )

        filter_condition = None
        if page_filter is not None:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=page_filter)
                    )
                ]
            )

        results = vector_store.similarity_search(
            query,
            k=3,
            filter=filter_condition
        )

        if not results:
            return "ผลการค้นหา: ไม่พบข้อมูลในเอกสารตามเงื่อนไขที่ระบุ"

        response_text = ""
        for i, doc in enumerate(results):
            page = doc.metadata.get("page", "unknown")
            response_text += f"\n -- ข้อมูลที่ {i+1} (หน้า {page}) : {doc.page_content[:150]}..."

        return response_text

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"

def start_agent():
    print("Starting Legal AI Agent...")

    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
        temperature=0,
    )

    tools = [search_legal_documents]
    memory = MemorySaver()

    agent_graph = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory
    )

    config = {"configurable": {"thread_id": "legal_ai_agent"}}

    print("Agent Ready! ลองพิมพ์: 'ช่วยต้นหาเรื่องสิงปฏิกูลในหน้า 69 ให้หน่อย'")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in agent_graph.stream(inputs, config=config, stream_mode="values"):
            message = event["messages"][-1]
            if message.type == "ai" and not message.tool_calls:
                print(f"AI: {message.content}")

        print("-" * 50)

if __name__ == "__main__":
    start_agent()
        