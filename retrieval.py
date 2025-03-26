# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
# import nltk
# import numpy as np
# import os
# import json
# nltk.download('punkt')
# nltk.download('punkt_tab')
# from nltk.tokenize import word_tokenize

# DATA_PATH = "./data"
# QA_FILE = f"{DATA_PATH}/QA_set/easy_single.json"
# ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"

# # Dense retrieval
# def dense_retrieval_subqueries(queries, model, faiss_index, passages, chunk_ids, top_k=5):
#     """
#     Perform dense retrieval for each sub-query and aggregate results.
    
#     Args:
#       queries (str or list): A single query or list of sub-queries.
#       model: The SentenceTransformer model.
#       faiss_index: A pre-built FAISS index.
#       passages: List of passages.
#       chunk_ids: List of chunk IDs corresponding to passages.
#       top_k (int): Number of results to retrieve per query.
      
#     Returns:
#       all_results (list): A list of dictionaries with keys: 'sub_query', 'chunk_id', 'passage', 'score'
#     """
#     # If a single query is passed as a string, wrap it in a list
#     if isinstance(queries, str):
#         queries = [queries]
    
#     all_results = []
#     for query in queries:
#         # Encode the sub-query
#         query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
#         # Retrieve top-k using FAISS
#         distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
#         # For each result, capture sub-query info along with chunk id, passage and score
#         results = [
#             {
#                 "sub_query": query,
#                 "chunk_id": chunk_ids[i],
#                 "passage": passages[i],
#                 "score": distances[0][j]
#             }
#             for j, i in enumerate(indices[0])
#         ]
#         all_results.extend(results)
#     return all_results

# # tfidf sparse retrieval
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Loader that returns both passages and chunk IDs
# def load_passages_and_chunk_ids():
#     if not os.path.exists(ALL_CHUNKS_FILE):
#         print(f"File not found: {ALL_CHUNKS_FILE}")
#         return [], []
#     with open(ALL_CHUNKS_FILE, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     passages = [entry["passage"] for entry in data if "passage" in entry]
#     chunk_ids = [entry["chunk_id"] for entry in data if "chunk_id" in entry]
#     print(f"Loaded {len(passages)} passages with chunk IDs")
#     return passages, chunk_ids

# # Load passages and chunk IDs
# passages, chunk_ids = load_passages_and_chunk_ids()

# # Build the TF-IDF matrix on your passages
# vectorizer = TfidfVectorizer(stop_words='english')
# doc_matrix = vectorizer.fit_transform(passages)

# def tfidf_retrieval(query, vectorizer, doc_matrix, passages, chunk_ids, top_k=5):
#     query_vec = vectorizer.transform([query])
#     # Compute cosine similarities between query and all documents
#     cosine_similarities = (doc_matrix * query_vec.T).toarray().flatten()
#     sorted_indices = np.argsort(cosine_similarities)[::-1][:top_k]
#     # Prepare a list of result dictionaries
#     results = [
#         {
#             "chunk_id": chunk_ids[i],
#             "passage": passages[i],
#             "score": cosine_similarities[i]
#         }
#         for i in sorted_indices
#     ]
#     return results, cosine_similarities[sorted_indices], sorted_indices

# # BM25 sparse retrieval
# # Loader that returns both passages and chunk IDs
# def load_passages_and_chunk_ids_2():
#     if not os.path.exists(ALL_CHUNKS_FILE):
#         print(f"File not found: {ALL_CHUNKS_FILE}")
#         return [], []
#     with open(ALL_CHUNKS_FILE, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     passages = [entry["passage"] for entry in data if "passage" in entry]
#     chunk_ids = [entry["chunk_id"] for entry in data if "chunk_id" in entry]
#     print(f"Loaded {len(passages)} passages with chunk IDs")
#     return passages, chunk_ids

# # Load passages and chunk IDs
# passages2, chunk_ids2 = load_passages_and_chunk_ids_2()

# # Tokenize the passages for BM25
# tokenized_passages2 = [word_tokenize(doc.lower()) for doc in passages2]

# # Build the BM25 index
# bm25 = BM25Okapi(tokenized_passages2)

# def bm25_retrieval_subqueries(queries, bm25, passages, chunk_ids, top_k=5):
#     """
#     Perform BM25 retrieval for each sub-query and aggregate results.
    
#     Args:
#       queries (str or list): A single query or a list of sub-queries.
#       bm25: The BM25Okapi instance.
#       passages: List of passage texts.
#       chunk_ids: List of chunk IDs corresponding to the passages.
#       top_k (int): Number of top results to retrieve per sub-query.
      
#     Returns:
#       all_results (list): A list of dictionaries with keys:
#         - 'sub_query': the sub-question used
#         - 'chunk_id': the identifier of the passage
#         - 'passage': the passage text
#         - 'score': the BM25 score
#     """
#     if isinstance(queries, str):
#         queries = [queries]
    
#     all_results = []
#     for query in queries:
#         tokenized_query = word_tokenize(query.lower())
#         scores = bm25.get_scores(tokenized_query)
#         sorted_indices = np.argsort(scores)[::-1][:top_k]
#         results = [
#             {
#                 "sub_query": query,
#                 "chunk_id": chunk_ids[i],
#                 "passage": passages[i],
#                 "score": scores[i]
#             }
#             for i in sorted_indices
#         ]
#         all_results.extend(results)
#     return all_results

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