"""
FastAPI wrapper around the LangGraph agent.
Run locally:
    export OPENAI_API_KEY=...
    uvicorn app.api:app --reload
"""
from __future__ import annotations
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from .agent_graph import GRAPH
from dotenv import load_dotenv

load_dotenv()

# FastAPI application for the ReadCoach chat interface
app = FastAPI(title="ReadCoach API", version="0.1.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """Serve the main chat UI."""
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    thread_id: str = Field(..., description="Stable id per user (or device) to preserve memory")
    message: str  # User's message to the AI librarian

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    """Main chat endpoint that processes user messages through the LangGraph agent.
    
    The thread_id enables persistent conversations - the agent remembers previous
    messages and user profile within the same thread.
    """
    # Configure LangGraph with thread_id for conversation persistence
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # Invoke the agent graph with the user's message
    # The graph will:
    # 1. Process the message through the LLM
    # 2. Call tools if needed (search, recommend, profile management)
    # 3. Generate a final response with book recommendations and explanations
    result = GRAPH.invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    
    # Extract the final assistant response
    last = result["messages"][-1]
    
    return {
        "thread_id": req.thread_id,
        "answer": getattr(last, "content", ""),  # The AI librarian's response
    }