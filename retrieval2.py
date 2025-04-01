import os
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# -----------------------------
# Config paths
# -----------------------------
DATA_PATH = "./data"
ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"
# -----------------------------
# Utility function to load passages
# -----------------------------
def load_passages_and_chunk_ids():
    if not os.path.exists(ALL_CHUNKS_FILE):
        raise FileNotFoundError(f"Missing file: {ALL_CHUNKS_FILE}")
    with open(ALL_CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    passages = [entry["passage"] for entry in data if "passage" in entry]
    chunk_ids = [entry["chunk_id"] for entry in data if "chunk_id" in entry]
    print(f"Loaded {len(passages)} passages with chunk IDs")
    return passages, chunk_ids
# -----------------------------
# TF-IDF Retrieval
# -----------------------------
def build_tfidf_index(passages):
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_matrix = vectorizer.fit_transform(passages)
    return vectorizer, doc_matrix

def tfidf_retrieval(query, vectorizer, doc_matrix, passages, chunk_ids, top_k=5):
    query_vec = vectorizer.transform([query])
    cosine_similarities = (doc_matrix @ query_vec.T).toarray().flatten()
    sorted_indices = np.argsort(cosine_similarities)[::-1][:top_k]
    return [
        {
            "chunk_id": chunk_ids[i],
            "passage": passages[i],
            "score": cosine_similarities[i]
        }
        for i in sorted_indices
    ]
# -----------------------------
# BM25 Retrieval
# -----------------------------
def build_bm25_index(passages):
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    return BM25Okapi(tokenized_passages)

def bm25_retrieval_subqueries(queries, bm25, passages, chunk_ids, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    for query in queries:
        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        results.extend([
            {
                "sub_query": query,
                "chunk_id": chunk_ids[i],
                "passage": passages[i],
                "score": scores[i]
            } for i in sorted_indices
        ])
    return results

# -----------------------------
# Dense Retrieval
# -----------------------------
def dense_retrieval_subqueries(queries, model, faiss_index, passages, chunk_ids, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    for query in queries:
        query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = faiss_index.search(np.expand_dims(query_emb, axis=0), top_k)
        results.extend([
            {
                "sub_query": query,
                "chunk_id": chunk_ids[i],
                "passage": passages[i],
                "score": distances[0][j]
            } for j, i in enumerate(indices[0])
        ])
    return results
# -----------------------------
# Hybrid retrieval
# -----------------------------
def hybrid_retrieval(
    queries,
    sparse_model,
    dense_model,
    passages,
    chunk_ids,
    sparse_retrieval_func,
    top_k=5,
    intermediate_k=1000
):
    if isinstance(queries, str):
        queries = [queries]
    
    intermediate_k = min(intermediate_k, len(passages))
    all_results = []
    for query in queries:
        # Sparse retrieval
        if isinstance(sparse_model, tuple):
            vectorizer, doc_matrix = sparse_model
            candidates = sparse_retrieval_func(query, vectorizer, doc_matrix, passages, chunk_ids, top_k=intermediate_k)
        else:
            candidates = sparse_retrieval_func(query, sparse_model, passages, chunk_ids, top_k=intermediate_k)
        
        # Dense re-ranking
        query_emb = dense_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        candidate_passages = [c["passage"] for c in candidates]
        candidate_embs = dense_model.encode(candidate_passages, convert_to_numpy=True, normalize_embeddings=True)
        similarities = candidate_embs @ query_emb
        
        # Rerank
        sorted_idxs = np.argsort(similarities)[::-1][:top_k]
        for idx in sorted_idxs:
            candidate = candidates[idx]
            all_results.append({
                "sub_query": query,
                "chunk_id": candidate["chunk_id"],
                "passage": candidate["passage"],
                "sparse_score": float(candidate["score"]),
                "dense_score": float(similarities[idx])
            })
    
    return sorted(all_results, key=lambda x: x["dense_score"], reverse=True)[:top_k]