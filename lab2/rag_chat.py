#version 1.0

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
VECTOR_DB_PATH = "./chroma_db_data"

def start_chat():
    print("Loading Vector Database...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH, 
        embedding_function=embedding_function
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    system_prompt = (
        "คุณเป็นผู้ช่วยทนายความที่เก่งกาจและแม่นยำ "
        "หน้าที่ของคุณคือตอบคำถามโดยใช้ข้อมูลจาก Context ที่ให้มาด้านล่างนี้เท่านั้น "
        "ถ้าข้อมูลใน Context ไม่เพียงพอที่จะตอบ ให้ตอบว่า 'ขออภัย ไม่พบข้อมูลดังกล่าวในเอกสาร'"
        "ห้ามกุเรื่องขึ้นมาเองเด็ดขาด "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
     
    print("AI Legal Assistant Ready!")
    print("-" * 50)

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        if not query.strip():
            continue

        print("Searching & Thinking...")
        
        response = rag_chain.invoke({"input": query})

        print(f"AI: {response['answer']}")
        print("Sources Used: ")
        unique_sources = set()
        for doc in response['context']:
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown Document')
            unique_sources.add(f"{os.path.basename(source)}, Page {page}")

        for source in unique_sources:
            print(source)
        
        print("-" * 50)

if __name__ == "__main__":
    start_chat()

            