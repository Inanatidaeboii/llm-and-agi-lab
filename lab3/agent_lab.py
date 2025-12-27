import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_classic.tools.retriever import create_retriever_tool

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

VECTOR_DB_PATH = "./chroma_db_data"

@tool
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        import numexpr
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def get_tools():
    tools = []

    tools.append(calculator)
    search = DuckDuckGoSearchResults()
    search_tool = search.as_tool(
        name="internet_search",
        description="Search for lastest news, stock prices, or current events from the internet"
    )
    tools.append(search_tool)

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

    return tools

def start_modern_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    tools = get_tools()

    agent_graph = create_react_agent(llm,tools)

    print("\n"+"="*50)
    print("Modern Agent Ready!")
    print("Capabilities: [Math] [Internet Search] [Internal Policy Search]")
    print("=" * 50 + "\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        if not query.strip():
            continue

        print("Searching & Thinking...")
        
        inputs = {"messages":[HumanMessage(content=query)]}
        
        try:
            for chunk in agent_graph.stream(inputs, stream_mode="values"):
                message = chunk["messages"][-1]
                if message.type == "ai" and message.text:
                    if not message.tool_calls:
                        print(f"AI: {message.text}")
                    else:
                        for tool_call in message.tool_calls:
                            print(f"[Action] Calling Tool : {tool_call['name']} with {tool_call['args']}")

                elif message.type == "tool":
                    print(f"[Observation] Tool Results: {message.content[:200]}...")

            print ("=" * 50 + "\n")
        
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    start_modern_agent()

        
ฟเ