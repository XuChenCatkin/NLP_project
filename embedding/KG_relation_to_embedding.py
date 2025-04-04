import os
import json
import numpy as np
import faiss
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
EMBEDDING_PATH = "./embedding"
DATA_PATH = "./data"
KG_PATH = f"{DATA_PATH}/KG_result_cleaned.json"


# mpnet_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# -----------------------------
# Data Loading Functions
# -----------------------------


def load_kg():
    relation_to_kgid_map = []
    kg_relations = []
    
    if not os.path.exists(KG_PATH):
        print(f"KG file not found: {KG_PATH}")
        return []
    with open(KG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i in range(len(data)):
        relations = data[f'{i+1}']['relations']
        for relation in relations:
            relation = relation.replace("|", " ")
            relation_to_kgid_map.append(i+1)
            kg_relations.append(relation)

    return kg_relations, relation_to_kgid_map

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
    parent_dir = os.path.dirname(filename)
    os.makedirs(parent_dir, exist_ok=True)
    faiss.write_index(index, filename)
    print(f"FAISS index stored: {filename}")


# -----------------------------
# Unified Passage Processing
# -----------------------------

def process_all_KG_relations(model, model_name):
    print(f"\nProcessing unified HP embeddings for model: {model_name}")
    passages,map = load_kg()
    if not passages:
        print("No passages to embed.")
        return
    embeddings = embed_texts(passages, model)
    os.makedirs(EMBEDDING_PATH, exist_ok=True)
    index_path = f"{EMBEDDING_PATH}/{model_name}/hp_kg_{model_name}.index"
    #emb_path = f"{EMBEDDING_PATH}/hp_all_{model_name}_embeddings.npy"
    store_faiss_index(embeddings, index_path)
    #np.save(emb_path, embeddings)
    print(f"Embeddings and FAISS index saved for {model_name}")





# -----------------------------
# Run Model
# -----------------------------

if __name__ == "__main__":
    MODEL_NAME = "BAAI/bge-base-en-v1.5_finetuned"
    # Load models on GPU
    bge_model = SentenceTransformer("CatkinChen/BAAI_bge-base-en-v1.5_retrieval_finetuned_v1", device=device)

    # Process unified passages for each model
    process_all_KG_relations(bge_model, MODEL_NAME)
    #process_all_hp_passages(mpnet_model, "mpnet")
    
    # # Process QA subqueries for each model
    # for file in qa_files:
    #     process_qa_subqueries(bge_model, MODEL_NAME, file)
    #process_qa_subqueries(mpnet_model, "mpnet")

    # MODEL_TAG = "dpr"
    
    # # Load DPR Context Encoder and its Tokenizer for passage encoding
    # ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    # dpr_context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
    
    # # Optionally, load DPR Question Encoder if needed for query encoding:
    # qs_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    # dpr_question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    
    # # Process unified passages (passage embeddings) using DPR Context Encoder
    # process_all_hp_passages_dpr(dpr_context_model, ctx_tokenizer, MODEL_TAG)
    
    # # Process QA subqueries using DPR Context Encoder
    # for file in qa_files:
    #     process_qa_subqueries_dpr(dpr_question_model, qs_tokenizer, MODEL_TAG, file)