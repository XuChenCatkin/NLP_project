# src/config.py
import os

# --- Dynamically find the project root ---
# This makes all paths work correctly whether run from notebooks/, src/, or the root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Base Paths ---
# All paths are now built from the project root.
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "embedding", "BAAI", "bge-base-en-v1.5_finetuned")

# --- Corpus File ---
CORPUS_FILE = os.path.join(DATA_PATH, "chunked_text_all_together_cleaned.json")

# --- QA & Query Files ---
QA_EASY_FILE = os.path.join(DATA_PATH, "QA_set", "easy_single_labeled.json")

# --- Faiss Index Paths ---
CORPUS_FAISS_INDEX = os.path.join(EMBEDDING_PATH, "hp_all_BAAI", "bge-base-en-v1.5_finetuned.index")
QUERY_FAISS_INDEX = os.path.join(EMBEDDING_PATH, "easy_single_labeled_embeddings.index")