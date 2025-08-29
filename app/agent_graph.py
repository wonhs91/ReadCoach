"""
LangGraph agent that chats with users and calls tools for recommendations.
"""
from __future__ import annotations
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

from .tools import get_book_details, get_book_recommendation, set_profile
from .config import OPENAI_MODEL

load_dotenv()

# System prompt that defines the AI librarian's personality and behavior
SYSTEM_PROMPT = SystemMessage(content=(
    "You are ReadCoach, a friendly school library assistant. "
    "Your goal is to help students discover books they'll love and to increase reading. "
    "Explain why a book fits the request, "
    "and use tools when you need facts from the catalog or when composing recommendations. "
    "Always offer to save or use a reading profile (age, favorite genres)."
))

def build_graph() -> Any:
    """Build the LangGraph conversation agent with tool-calling capabilities.
    
    Flow: User message -> Assistant (LLM) -> Tools (if needed) -> Assistant -> Response
    The agent can call multiple tools in sequence to gather information before responding.
    """
    # Define available tools for the LLM to call
    tools = [get_book_details, get_book_recommendation, set_profile]
    
    # Create LLM with tool-calling capabilities (temperature=0 for consistent recommendations)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0).bind_tools(tools)

    def assistant(state: MessagesState, config=None):
        """Main LLM node that processes messages and decides whether to call tools."""
        # Prepend system prompt to conversation history
        messages = [SYSTEM_PROMPT] + state["messages"]
        
        # LLM generates response, potentially with tool calls
        resp = llm.invoke(messages, config=config)
        
        # Add LLM response to conversation state
        return {"messages": [resp]}

    # Tool execution node (handles all tool calls from LLM)
    tool_node = ToolNode(tools)

    # Build the conversation graph
    builder = StateGraph(MessagesState)  # MessagesState tracks conversation history
    builder.add_node("assistant", assistant)  # LLM reasoning and response generation
    builder.add_node("tools", tool_node)     # Tool execution (search, recommend, etc.)

    # Define conversation flow
    builder.add_edge(START, "assistant")  # Always start with the assistant
    
    # Conditional routing: if LLM made tool calls, execute them; otherwise end conversation
    builder.add_conditional_edges(
        "assistant",
        tools_condition,  # Built-in function that checks for tool_calls in last message
        {"tools": "tools", "__end__": END},  # Route to tools or end
    )
    
    # After tools execute, return to assistant for final response
    builder.add_edge("tools", "assistant")

    # Add persistence for multi-turn conversations and user profiles
    checkpointer = InMemorySaver()  # Stores conversation history by thread_id
    store = InMemoryStore()         # Stores user profiles and preferences
    
    # Compile the graph with persistence
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph

GRAPH = build_graph()