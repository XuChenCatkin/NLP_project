import os
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import faiss

from KG_retrieval import *

# -----------------------------
# Config paths
# -----------------------------
DATA_PATH = "./data"
ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"
print(f"Loading passages from {ALL_CHUNKS_FILE}")
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

def query_embed_search(query, all_queries_list, index):
    try:
        # Find the position (index) of the query in the list of all queries
        position = all_queries_list.index(query)
        # print(f'Found Position: {position}')
        # Reconstruct and return the vector at the given position in the index
        return index.reconstruct(position)
    except ValueError:
        raise ValueError(f"Query '{query}' not found in the list of all queries.")
    

def dense_retrieval_subqueries(queries, all_queries_list, sub_queries_index, faiss_index, passages, chunk_ids, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    for query in queries:
        #query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        query_emb = query_embed_search(query, all_queries_list, sub_queries_index)
        query_emb = query_emb.reshape(1, -1)  # Reshapes to (1, d)
        distances, indices = faiss_index.search(query_emb, top_k)
        results.extend([
        {
            "sub_query": query,
            "chunk_id": chunk_ids[i],
            "passage": passages[i],
            "score": float(distances[0][j])
        } for j, i in enumerate(indices[0])
        ])
    return results

def retrieve_all_subqueries(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)

    subquestion_list = []
    for i, item in enumerate(qa_data):
        tmp = item['sub_questions']
        for j, sub in enumerate(tmp):
            subquestion_list.append(sub)
    return subquestion_list


def dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index, all_chucks, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    for query in queries:
        #query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        query_emb = query_embed_search(query, all_queries_list, sub_queries_index)
        query_emb = query_emb.reshape(1, -1)  # Reshapes to (1, d)
        distances, indices = faiss_index.search(query_emb, top_k)
        results.extend([ all_chucks[i] for _, i in enumerate(indices[0]) ])
    return results

def KG_dense_retrieval(queries, all_queries_list, sub_queries_index, faiss_index_kg, faiss_index_chunk, all_chucks, relation_to_kgid_map, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    kg_ids = []
    for query in queries:
        #query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        query_emb = query_embed_search(query, all_queries_list, sub_queries_index)
        query_emb = query_emb.reshape(1, -1)  # Reshapes to (1, d)
        distances, indices = faiss_index_kg.search(query_emb,1000)
        entities = process_gpt(query)
        shared_kg = find_chunk_id(entities)
        relation_rank_index = indices[0]
        for i in relation_rank_index:
            if relation_to_kgid_map[i] in shared_kg:
                kg_ids.append(relation_to_kgid_map[i])
        if len(kg_ids) == 0:
            print('KG method failed')
            print('return the dense retrieval')
            return dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index_chunk, all_chucks, top_k=5)
        else:
            chunk_id_list = []
            for kg_id in kg_ids:
                chunk_id_list.extend([i for i in range(5*kg_id, 5*kg_id+5)])
                chunk_id_list = list(set(chunk_id_list))
            return dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index_chunk[chunk_id_list], all_chucks[chunk_id_list], top_k=5)

if __name__ == "__main__":
    data = {
        "question": "On which street do the Dursleys live at the beginning of the story?",
        "answer": "They live at Number Four, Privet Drive.",
        "list of reference": [
            {
                "ref_id": 1,
                "passage": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors.",
                "book": 1,
                "chapter": 1
            }
        ],
        "id": 1,
        "question_variants": "On which street do the Dursleys live at the beginning of the story?",
        "sub_questions": [
            "On which street do the Dursleys live at the beginning of the story?"
        ],
        "category": "easy_single_labeled"
    }
    EMBEDDING_PATH = "./embedding"
    DATA_PATH = "./data"
    KG_PATH = f"{DATA_PATH}/KG_result_cleaned.json"
    relation_to_kgid_map = []
    with open(KG_PATH, "r", encoding="utf-8") as f:
        KGs = json.load(f)
    for i in range(len(KGs)):
        relations = KGs[f'{i+1}']['relations']
        for relation in relations:
            relation = relation.replace("|", " ")
            relation_to_kgid_map.append(i+1)

    EASY_INDEX = faiss.read_index(f"embedding/BAAI/bge-base-en-v1.5_finetuned/easy_single_labeled_embeddings.index")
    EASY_ALL_SUB = retrieve_all_subqueries(f"{DATA_PATH}/QA_set/easy_single_labeled.json")
    CORPUS_EMBEDDING = faiss.read_index('embedding/BAAI/bge-base-en-v1.5_finetuned/hp_all_BAAI/bge-base-en-v1.5_finetuned.index')
    KG_EMBEDDING = faiss.read_index('embedding/BAAI/bge-base-en-v1.5_finetuned/hp_all_BAAI/bge-base-en-v1.5_finetuned.index')
    CORPUS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"
    with open(CORPUS_FILE, 'r') as f:
        CORPUS_DATA = json.load(f)
    # result = dense_retrieval_subqueries_for_finetune(data['sub_questions'], EASY_ALL_SUB, EASY_INDEX, CORPUS_EMBEDDING,CORPUS_DATA , top_k=5)
    result = KG_dense_retrieval(data['sub_questions'], EASY_ALL_SUB, EASY_INDEX, KG_EMBEDDING, CORPUS_EMBEDDING, CORPUS_DATA, relation_to_kgid_map, top_k=5)
    print(result)