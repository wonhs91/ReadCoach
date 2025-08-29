"""
App configuration and common paths.
Edit ARTIFACTS_DIR if you want to store indexes elsewhere.
"""
from __future__ import annotations
import os
from pathlib import Path

# Core paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
FAISS_DIR = ARTIFACTS_DIR / "faiss_index"
ITEM_SIM_PATH = ARTIFACTS_DIR / "item_sim.json"
CATALOG_CACHE = ARTIFACTS_DIR / "catalog_cache.json"

# Models (Option A – OpenAI)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# Safety / filters
DEFAULT_MIN_AGE = int(os.getenv("DEFAULT_MIN_AGE", "6"))
DEFAULT_MAX_AGE = int(os.getenv("DEFAULT_MAX_AGE", "18"))