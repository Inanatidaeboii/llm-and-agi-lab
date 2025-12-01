import tiktoken
from google import genai
from google.genai import types
import pydantic
import dotenv
import os
from typing import List, Optional

dotenv.load_dotenv()

PRICE_PER_1M_INPUT = 5.0
PRICE_PER_1M_OUTPUT = 15.0
USD_TO_THB = 34.0

class Message(pydantic.BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

    def count_tokens(self) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(self.content))

class ChatSession:
    def __init__(self):
        self.messages: List[Message] = []
        self.max_tokens = 4000
        self.total_cost_thb = 0.0

    def calculate_current_tokens(self) -> int:
        return sum(message.count_tokens() for message in self.messages)

    def add_message(self, message: Message):
        current_tokens = self.calculate_current_tokens()
        new_msg_tokens = message.count_tokens()

        while current_tokens + new_msg_tokens > self.max_tokens:
            if not self.messages:
                break

            if self.messages[0].role == "system":
                if len(self.messages) == 1:
                    removed = self.messages.pop(1)
                
                else:
                    break
            else:
                removed = self.messages.pop(0)

            print(f"Memory Full! Removing old message from '{removed.role}' ({removed.count_tokens()} tokens)...")
            current_tokens = self.calculate_current_tokens()
        
        self.messages.append(message)

        input_cost = (new_msg_tokens/1_000_000) * PRICE_PER_1M_INPUT * USD_TO_THB
        self.total_cost_thb += input_cost

    def update_cost_from_output(self, output: str):
        encoding = tiktoken.get_encoding("cl100k_base")
        output_tokens = len(encoding.encode(output))
        output_cost = (output_tokens/1_000_000) * PRICE_PER_1M_OUTPUT * USD_TO_THB
        self.total_cost_thb += output_cost

        return output_tokens

if __name__ == "__main__":
    def convert_to_gemini_format(messages):
        gemini_history = []
        for msg in messages:
            role = 'model' if msg.role == 'assistant' else 'user'
            gemini_history.append({
                "role": role,
                "parts": [
                    types.Part.from_text(text=msg.content)
                ]
            })
        return gemini_history
    
    def run_chat():
        chat_session = ChatSession()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        print("Welcome to My Chatbot!\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            message = Message(role='user', content=user_input)
            chat_session.add_message(message)

            gemini_payload = convert_to_gemini_format(chat_session.messages)

            try:
                response = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_payload)
                response_message = Message(role='assistant', content=response.text)
                chat_session.add_message(response_message)
                chat_session.update_cost_from_output(response.text)
                print(f"Assistant: {response.text}")
                print(f"[Stats] Tokens: {chat_session.calculate_current_tokens()}/{chat_session.max_tokens} | Cost: {chat_session.total_cost_thb:.4f} THB")
                print("-" * 30)

            except Exception as e:
                print(f"Error: {str(e)}")
        
    run_chat()
