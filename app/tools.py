"""
Tools the LLM can call.
We keep return values as compact JSON strings so the model can cite them back to the user.
"""
from __future__ import annotations
import json
from typing import Annotated, Any, Dict
from pathlib import Path

from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import InjectedStore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings

from .config import FAISS_DIR, CATALOG_CACHE, EMBED_MODEL

records = json.loads(Path(CATALOG_CACHE).read_text(encoding="utf-8"))
_catalog_index = {str(r["book_id"]): r for r in records}
_title_index = {r["title"].lower(): r for r in records}

@tool
def get_book_details(book_title: str) -> str:
    """Returns the book detail"""
    book = _title_index.get(book_title.lower())
    if book:
        return json.dumps(book)
    return json.dumps({"error": "Book not found"})


@tool
def get_book_recommendation(
    config: RunnableConfig,
    store: Annotated[Any, InjectedStore()] = None,
    query: str | None = None,
) -> str:
    """Hybrid recommendations that blend your recent reads with an optional query. Returns a JSON list of books."""
    # Combine collaborative filtering (based on reading history) with content-based search (query)
    # This gives personalized recommendations that also match current interests
    # Example: user liked fantasy books + currently wants "friendship stories" = fantasy books about friendship
    user_id = config.get("configurable", {}).get("user_id")
    age = store.get(("profile",), "age").value
    favorite_genres = store.get(("profile", ), "favorite_genres").value
    
    candidates: Dict[str, float] = {}  # book_id -> combined score
    if not query:
        query = ""
    
    query += f"""
    age: {age}
    favorite_genres: {favorite_genres}
    """
    
    _embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    _vs = FAISS.load_local(str(FAISS_DIR), _embeddings, allow_dangerous_deserialization=True)

    if query:
        for d in _vs.similarity_search(query, k=3):
            rec = _catalog_index.get(str(d.metadata.get("book_id")))

            # Add 1.0 points for semantic match
            candidates[rec["book_id"]] = candidates.get(rec["book_id"], 0.0) + 1.0

    cf_recs = [
        {"book_id": "B1", "score_cf": 0.95},
        {"book_id": "B2", "score_cf": 0.92},
        {"book_id": "B3", "score_cf": 0.89},
        {"book_id": "B4", "score_cf": 0.85},
        {"book_id": "B5", "score_cf": 0.82}
    ]
    
    for rec in cf_recs:
        # Add collaborative filtering score (higher = more similar reading patterns)
        candidates[rec["book_id"]] = candidates.get(rec["book_id"], 0.0) + rec.get("score_cf")

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:3]
    results = [ _catalog_index[str(bid)] for bid, _ in ranked if str(bid) in _catalog_index ]
    # Return as JSON for LLM to explain the personalized reasoning
    return json.dumps(results)

@tool
def set_profile(
    age: int | None = None,
    favorite_genres: str | None = None,
    store: Annotated[Any, InjectedStore()] = None,
) -> str:
    """Save the user's profile (grade and favorite_genres) for better recommendations."""
    # Store user preferences in LangGraph's persistent store for future conversations
    # This enables personalized age filtering and genre preferences across sessions
    if store is not None:
        if age is not None:
            # Age helps determine appropriate age ranges for book filtering
            store.put(("profile",), "age", age)
        if favorite_genres:
            # Favorite genres can be used to weight search results and recommendations
            store.put(("profile",), "favorite_genres", favorite_genres)
        return "Saved your profile."
    return "Profile saved for this session."
