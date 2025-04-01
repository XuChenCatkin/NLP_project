import os
import json
import numpy as np
import faiss
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob

# Use CUDA if available
#device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths
EMBEDDING_PATH = "./embedding"
DATA_PATH = "./data"
ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"
QA_PATH = f"{DATA_PATH}/QA_set"
qa_files = glob.glob(f"{QA_PATH}/*.json")
qa_files = [file for file in qa_files if "_labeled" in file]

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models on GPU
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
# mpnet_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# -----------------------------
# Data Loading Functions
# -----------------------------

# Load all passages from the unified file
def load_all_passages():
    if not os.path.exists(ALL_CHUNKS_FILE):
        print(f"File not found: {ALL_CHUNKS_FILE}")
        return []
    with open(ALL_CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    passages = [entry["passage"] for entry in data if "passage" in entry]
    print(f"Total passages loaded: {len(passages)}")
    return passages

# Load QA questions from the QA file (if needed)
def load_qa_questions(file):
    if not os.path.exists(file):
        print(f"QA file not found: {file}")
        return []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data if "question" in item]
    print(f"Total QA questions loaded: {len(questions)} from {file}")
    return questions

# -----------------------------
# Embedding and FAISS Utilities
# -----------------------------

# Generate embeddings
def embed_texts(texts, model, batch_size=8):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_embeds = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_embeds)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.vstack(embeddings)

# Store FAISS index
def store_faiss_index(embeddings, filename):
    dim = embeddings.shape[1]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalised_embeddings = embeddings / norms
    # index = faiss.IndexFlatL2(dim)
    index = faiss.IndexFlatIP(dim)  # Use inner product for cosine similarity
    index.add(normalised_embeddings)
    faiss.write_index(index, filename)
    print(f"FAISS index stored: {filename}")

# -----------------------------
# Unified Passage Processing
# -----------------------------

def process_all_hp_passages(model, model_name):
    print(f"\nProcessing unified HP embeddings for model: {model_name}")
    passages = load_all_passages()
    if not passages:
        print("No passages to embed.")
        return
    embeddings = embed_texts(passages, model)
    os.makedirs(EMBEDDING_PATH, exist_ok=True)
    index_path = f"{EMBEDDING_PATH}/hp_all_{model_name}.index"
    #emb_path = f"{EMBEDDING_PATH}/hp_all_{model_name}_embeddings.npy"
    store_faiss_index(embeddings, index_path)
    #np.save(emb_path, embeddings)
    print(f"Embeddings and FAISS index saved for {model_name}")

# -----------------------------
# QA Subqueries Processing
# -----------------------------

# Load subqueries from the QA file
def load_qa_subqueries(file):
    if not os.path.exists(file):
        print(f"QA file not found: {file}")
        return []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract subqueries from the "sub_questions" field in each QA entry.
    sub_queries = []
    for item in data:
        if "sub_questions" in item:
            sub_queries.extend(item["sub_questions"])
    print(f"Total subqueries loaded: {len(sub_queries)}")
    return sub_queries

# Unified QA processing for subqueries
def process_qa_subqueries(model, model_name, file):
    print(f"\nProcessing QA embeddings for subqueries with model: {model_name}")
    sub_queries = load_qa_subqueries(file)
    if not sub_queries:
        print("No subqueries found.")
        return
    
    base_name = os.path.splitext(os.path.basename(file))[0]
    
    embeddings = embed_texts(sub_queries, model)
    index_path = f"{EMBEDDING_PATH}/{base_name}_embeddings.index"
    #emb_path = f"{EMBEDDING_PATH}/qa_subquery_embeddings_{model_name}.npy"
    store_faiss_index(embeddings, index_path)
    #np.save(emb_path, embeddings)
    print(f"QA subquery embeddings and FAISS index saved for {model_name}")

# -----------------------------
# Run for Both Models
# -----------------------------

if __name__ == "__main__":
    # Process unified passages for each model
    process_all_hp_passages(bge_model, "bge")
    #process_all_hp_passages(mpnet_model, "mpnet")
    
    # Process QA subqueries for each model
    for file in qa_files:
        process_qa_subqueries(bge_model, "bge", file)
    #process_qa_subqueries(mpnet_model, "mpnet")