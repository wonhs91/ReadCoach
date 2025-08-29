"""
Data prep utilities:
- Load catalog and borrow history from CSV.
- Build FAISS vector index from catalog metadata.
- Build simple item-item co-occurrence similarity from borrow history.
- Persist artifacts under artifacts/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import FAISS_DIR, ITEM_SIM_PATH, CATALOG_CACHE, EMBED_MODEL

def _normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize catalog data for consistent processing."""
    # Ensure required columns exist
    required = ["book_id", "title"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in catalog.csv")
    
    # Add missing optional columns with default values
    for col in ["authors", "summary", "subjects", "age_min", "age_max"]:
        if col not in df.columns:
            df[col] = None
    
    # Standardize data types
    df["book_id"] = df["book_id"].astype(str)  # Ensure string IDs for consistency
    df["title"] = df["title"].astype(str)
    
    # Handle age ranges with sensible defaults (6-18 years)
    df["age_min"] = pd.to_numeric(df["age_min"], errors="coerce").fillna(6).astype(int)
    df["age_max"] = pd.to_numeric(df["age_max"], errors="coerce").fillna(18).astype(int)
    
    # Fill missing text fields with empty strings
    for col in ["authors", "summary", "subjects"]:
        df[col] = df[col].fillna("")
    
    return df

def _normalize_borrows(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize borrow history data."""
    # Ensure required columns for collaborative filtering
    required = ["user_id", "book_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in borrows.csv")
    
    # Standardize IDs as strings for consistent matching
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)
    
    # Parse timestamps if available (for future temporal features)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    return df

def build_catalog_index(catalog_csv: Path, out_dir: Path = FAISS_DIR) -> None:
    """Build FAISS vector index from book catalog for semantic search."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(catalog_csv)
    df = _normalize_catalog(df)

    # Create documents for embedding: combine all searchable text fields
    docs: List[Document] = []
    for _, row in df.iterrows():
        # Concatenate title, authors, subjects, and summary for rich semantic search
        text = " | ".join([
            row["title"],
            str(row["authors"] or ""),
            str(row["subjects"] or ""),
            str(row["summary"] or ""),
        ])
        
        # Store book metadata for filtering and display
        docs.append(Document(
            page_content=text,  # Text to embed
            metadata={
                "book_id": row["book_id"],
                "title": row["title"],
                "authors": row["authors"],
                "subjects": row["subjects"],
                "age_min": int(row["age_min"]),
                "age_max": int(row["age_max"]),
            },
        ))

    # Generate embeddings and build FAISS index
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(docs, embeddings)  # Creates vector store with embeddings
    vs.save_local(str(out_dir))  # Persist to disk

    # Cache catalog as JSON for fast metadata lookup during recommendations
    with open(CATALOG_CACHE, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False)

    print(f"[OK] Saved FAISS index to {out_dir} and catalog cache to {CATALOG_CACHE}")

def build_item_item(borrows_csv: Path, out_path: Path = ITEM_SIM_PATH, top_k: int = 50) -> None:
    """Build item-item collaborative filtering similarity matrix from borrow history.
    
    Uses cosine similarity normalized by popularity: sim(i,j) = co_borrows(i,j) / sqrt(pop(i) * pop(j))
    This prevents popular books from dominating recommendations.
    """
    df = pd.read_csv(borrows_csv)
    df = _normalize_borrows(df)

    # Group books by user to find co-occurrence patterns
    user_groups = df.groupby("user_id")["book_id"].apply(list)
    co_counts: Dict[str, Dict[str, int]] = {}  # Count of users who borrowed both books
    pop: Dict[str, int] = {}  # Popularity count per book

    # Calculate co-occurrence and popularity from user reading patterns
    for books in user_groups:
        unique_books = list(dict.fromkeys(books))  # Remove duplicates per user
        
        # Count all pairwise co-occurrences within this user's history
        for i in range(len(unique_books)):
            bi = unique_books[i]
            pop[bi] = pop.get(bi, 0) + 1  # Increment popularity
            
            # Count co-occurrences with all other books this user read
            for j in range(i + 1, len(unique_books)):
                bj = unique_books[j]
                # Initialize nested dictionaries if needed
                co_counts.setdefault(bi, {}).setdefault(bj, 0)
                co_counts.setdefault(bj, {}).setdefault(bi, 0)
                # Increment co-occurrence count (symmetric)
                co_counts[bi][bj] += 1
                co_counts[bj][bi] += 1

    # Calculate normalized similarity scores for each book
    item_sim: Dict[str, list] = {}
    for bi, neighbors in co_counts.items():
        sims = []
        for bj, cij in neighbors.items():
            # Cosine similarity normalized by popularity (prevents popular book bias)
            denom = np.sqrt(pop.get(bi, 1) * pop.get(bj, 1))
            score = float(cij / denom) if denom > 0 else 0.0
            sims.append((bj, score))
        
        # Keep only top-k most similar books, sorted by similarity
        sims.sort(key=lambda x: x[1], reverse=True)
        item_sim[bi] = sims[:top_k]

    # Save similarity matrix to JSON for fast lookup during recommendations
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(item_sim, f)

    print(f"[OK] Saved item-item similarity to {out_path}")

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True, help="Path to catalog.csv")
    p.add_argument("--borrows", type=Path, required=True, help="Path to borrows.csv")
    p.add_argument("--out_dir", type=Path, default=None, help="Artifacts directory")
    args = p.parse_args()

    if args.out_dir:
        # Optional override for artifact locations
        from . import config as cfg
        cfg.ARTIFACTS_DIR = args.out_dir
        cfg.FAISS_DIR = args.out_dir / "faiss_index"
        cfg.ITEM_SIM_PATH = args.out_dir / "item_sim.json"
        cfg.CATALOG_CACHE = args.out_dir / "catalog_cache.json"

    build_catalog_index(args.catalog)
    build_item_item(args.borrows)

if __name__ == "__main__":
    cli()