import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
import uvicorn

app = FastAPI()

# --- Configuration ---
# In keys ko aapne Render ki settings mein daalna hai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class UltraAgent1:
    def __init__(self):
        # Professional Brain (GPT-4o)
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.2, 
            api_key=OPENAI_API_KEY
        )
        
        # Deep Research Tool
        self.search = TavilySearchResults(api_key=TAVILY_API_KEY)
        
        self.tools = [
            Tool(
                name="Research",
                func=self.search.run,
                description="Use this to find latest professional info on the web."
            )
        ]
        
        # Reasoning Engine
        self.agent = initialize_agent(
            self.tools, 
            self.llm, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True
        )

    def think_and_reply(self, user_input):
        system_instructions = (
            "Name: Ultra Agent 1. Developer: Muhammad Ayaan. "
            "Instruction: Research first, think deep, and answer professionally in the user's language."
        )
        return self.agent.run(input=f"{system_instructions}\n\nUser: {user_input}")

# Initialize
bot = UltraAgent1()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Ultra Agent 1 is Active", "developer": "Muhammad Ayaan"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        reply = bot.think_and_reply(request.message)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)