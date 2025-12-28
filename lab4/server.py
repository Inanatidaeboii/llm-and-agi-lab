import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

VECTOR_DB_PATH = "./chroma_db"
agent_graph = None

@tool
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        import numexpr
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def setup_agent():
    print("Setting up Agent...")

    tools = [calculator]

    search = DuckDuckGoSearchRun()
    tools.append(search.as_tool(name="internet_search", description="Search for lastest news, stock prices, or current events from the internet"))

    if os.path.exists(VECTOR_DB_PATH):
        print("Loading Vector Database...")
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_function
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        rag_tool = create_retriever_tool(
            retriever,
            name="internal_policy_search",
            description="Search for internal policies and procedures"
        )

        tools.append(rag_tool)
    else:
        print("Vector Database not found. Skipping internal policy search tool.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    system_prompt = (
        "คุณเป็นผู้ช่วยทางการเงินและกฎหมายที่เชี่ยวชาญ "
        "ต้องตอบคำถามให้กระชับ ชัดเจน และอ้างอิงแหล่งที่มาเสมอ "
        "หากมีการคำนวณตัวเลข ต้องใช้เครื่องมือ calculator ห้ามคิดเอง"
    )
    agent_graph = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_prompt,
    )
    return agent_graph

app = FastAPI(
    title="Duck the Financial Agent",
    description="API สำหรับให้บริการ AI Agent ที่สามารถค้นหาข้อมูล, อ่านเอกสาร, และคำนวณเงินได้",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    user_id: str = "guest"

class StepLog(BaseModel):
    step_type: str
    details: str

class AgentResponse(BaseModel):
    answer: str
    process_logs: list[StepLog] = []

@app.on_event("startup")
async def startup_event():
    global agent_graph
    try:
        agent_graph = setup_agent()
        print("Server Ready! Agent Loaded.")
    except Exception as e:
        print(f"Error setting up agent: {str(e)}")

@app.post("/api/chat", response_model=AgentResponse)
async def chat_endpoint(request: QueryRequest):
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent not loaded")

    print(f"Request from {request.user_id}: {request.query}")
    
    final_answer = ""
    logs = []

    try:
        inputs = {"messages": [HumanMessage(content=request.query)]}

        for chunk in agent_graph.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if message.type == "ai":
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        log_msg = f"Calling Tool: {tool_call['name']} (Args: {tool_call['args']})"
                        logs.append(StepLog(step_type="tool_call", details=log_msg))
                        print(f"   [Action] {log_msg}")
                elif message.text:
                    final_answer = message.text

            elif message.type == "tool":
                log_msg = f"Tool Result: {message.text[:200]}..."
                logs.append(StepLog(step_type="tool_result", details=log_msg))
                print(f"   [Observation] {log_msg}")
        
        return AgentResponse(
            answer=final_answer,
            process_logs=logs
        )

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
