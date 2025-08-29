"""
Hybrid recommendation utilities combining:
- vector similarity over catalog metadata (content-based)
- item-item collaborative filtering from co-borrows (implicit feedback)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .config import FAISS_DIR, ITEM_SIM_PATH, CATALOG_CACHE, EMBED_MODEL, DEFAULT_MIN_AGE, DEFAULT_MAX_AGE

# Global state for lazy loading of recommendation artifacts
_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
_vs: Optional[FAISS] = None  # FAISS vector store for semantic search
_item_sim: Dict[str, List[Tuple[str, float]]] = {}  # Item-item similarity matrix
_catalog_index: Dict[str, dict] = {}  # Fast book metadata lookup by ID


def _ensure_loaded():
    """Lazy load all recommendation artifacts on first use."""
    global _vs, _item_sim, _catalog_index
    
    # Load FAISS vector store for semantic search
    if _vs is None:
        _vs = FAISS.load_local(str(FAISS_DIR), _embeddings, allow_dangerous_deserialization=True)
    
    # Load collaborative filtering similarity matrix
    if not _item_sim:
        _item_sim = json.loads(Path(ITEM_SIM_PATH).read_text(encoding="utf-8"))
    
    # Load catalog metadata for fast book info lookup
    if not _catalog_index:
        records = json.loads(Path(CATALOG_CACHE).read_text(encoding="utf-8"))
        _catalog_index = {str(r["book_id"]): r for r in records}

def _age_ok(rec: dict, min_age: int | None, max_age: int | None) -> bool:
    """Check if a book's age range overlaps with the requested age range.
    
    Args:
        rec: Book record with age_min/age_max fields
        min_age: Minimum age requirement (None = no minimum)
        max_age: Maximum age requirement (None = no maximum)
    
    Returns:
        True if book's age range overlaps with requested range
    """
    if rec is None:
        return False
    
    # Get book's age range (with defaults)
    lo = int(rec.get("age_min", DEFAULT_MIN_AGE))
    hi = int(rec.get("age_max", DEFAULT_MAX_AGE))
    
    # No age restrictions = always OK
    if min_age is None and max_age is None:
        return True
    
    # Check for overlap between book's range [lo, hi] and requested range [min_age, max_age]
    if min_age is None:
        return lo <= max_age  # Book starts before max requested age
    if max_age is None:
        return hi >= min_age  # Book extends past min requested age
    
    # Both bounds specified: ranges must overlap
    return (lo <= max_age) and (hi >= min_age)

def vector_search(query: str, top_k: int = 10, min_age: int | None = None, max_age: int | None = None) -> List[dict]:
    """Content-based search using semantic similarity of book metadata.
    
    Args:
        query: Natural language search query (e.g., "fantasy adventure books")
        top_k: Number of results to return
        min_age/max_age: Age filtering bounds
    
    Returns:
        List of book records matching the query and age constraints
    """
    _ensure_loaded()
    
    # Search FAISS index with query embedding, overfetch to account for age filtering
    docs = _vs.similarity_search(query, k=max(50, top_k))
    
    # Filter results by age appropriateness and collect metadata
    out: List[dict] = []
    for d in docs:
        # Get full book record from cache using book_id from FAISS metadata
        rec = _catalog_index.get(str(d.metadata.get("book_id")))
        if rec and _age_ok(rec, min_age, max_age):
            out.append(rec)
        if len(out) >= top_k:
            break
    return out

def similar_to(book_id: str, top_k: int = 10, min_age: int | None = None, max_age: int | None = None) -> List[dict]:
    """Collaborative filtering: find books similar to a given book based on co-borrowing patterns.
    
    Args:
        book_id: ID of the seed book to find similarities for
        top_k: Number of similar books to return
        min_age/max_age: Age filtering bounds
    
    Returns:
        List of similar book records with collaborative filtering scores
    """
    _ensure_loaded()
    
    # Get precomputed similarity scores for this book
    sims = _item_sim.get(str(book_id), [])
    
    # Filter by age and collect book metadata with CF scores
    out: List[dict] = []
    for bj, score in sims[: max(50, top_k)]:  # Overfetch for age filtering
        rec = _catalog_index.get(str(bj))
        if rec and _age_ok(rec, min_age, max_age):
            # Add collaborative filtering score to book record
            out.append(rec | {"_score_cf": float(score)})
        if len(out) >= top_k:
            break
    return out

def hybrid_for_user(
    user_history: List[str],
    query: str | None = None,
    top_k: int = 10,
) -> List[dict]:
    """Hybrid recommendation combining content-based and collaborative filtering.
    
    Args:
        user_history: List of book IDs the user has previously read
        query: Optional text query to blend with user history
        top_k: Number of recommendations to return
        min_age/max_age: Age filtering bounds
    
    Returns:
        List of recommended books ranked by hybrid score
    """
    _ensure_loaded()
    candidates: Dict[str, float] = {}  # book_id -> combined score

    # Content-based component: if user provided a query, find semantically similar books
    if query:
        for rec in vector_search(query, top_k=top_k * 3, min_age=min_age, max_age=max_age):
            # Add 1.0 points for semantic match
            candidates[rec["book_id"]] = candidates.get(rec["book_id"], 0.0) + 1.0

    # Collaborative filtering component: find books similar to user's recent reads
    for seed in user_history[-5:]:  # Use last 5 books to avoid noise from old preferences
        for rec in similar_to(seed, top_k=top_k * 3, min_age=min_age, max_age=max_age):
            # Add collaborative filtering score (higher = more similar reading patterns)
            candidates[rec["book_id"]] = candidates.get(rec["book_id"], 0.0) + (rec.get("_score_cf") or 0.5)

    # Remove books the user has already read
    for b in set(user_history):
        candidates.pop(b, None)

    # Return top recommendations sorted by combined score
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[: top_k]
    return [ _catalog_index[str(bid)] for bid, _ in ranked if str(bid) in _catalog_index ]