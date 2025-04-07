import sys
import os
import json
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from evaluation.eval_utils import *
from generation.cohere_generation import *

# Set up path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#########################################
# 1. Retrieval Functions
#########################################

def tfidf_retrieval(query, config):
    """
    TF-IDF retrieval.
    config must include:
      - "vectorizer": a fitted TfidfVectorizer,
      - "doc_matrix": document-term matrix,
      - "passages": list of passages,
      - "chunk_ids": list of passage identifiers,
      - "top_k": number of top results.
    """
    vectorizer = config["vectorizer"]
    doc_matrix = config["doc_matrix"]
    passages = config["passages"]
    chunk_ids = config["chunk_ids"]
    top_k = config.get("top_k", 5)
    
    query_vec = vectorizer.transform([query])
    cosine_similarities = (doc_matrix @ query_vec.T).toarray().flatten()
    sorted_indices = np.argsort(cosine_similarities)[::-1][:top_k]
    
    results = [{"chunk_id": chunk_ids[i], "score": float(cosine_similarities[i])} 
               for i in sorted_indices]
    return results

def bm25_retrieval(query, config):
    """
    BM25 retrieval.
    config must include:
      - "bm25": a BM25Okapi object,
      - "passages": list of passages,
      - "chunk_ids": list of passage identifiers,
      - "top_k": number of top results.
    """
    bm25 = config["bm25"]
    passages = config["passages"]
    chunk_ids = config["chunk_ids"]
    top_k = config.get("top_k", 5)
    
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    
    results = [{"chunk_id": chunk_ids[i], "score": float(scores[i])} 
               for i in sorted_indices]
    return results

def dense_retrieval_subqueries(queries, config):

    if isinstance(queries, str):
        queries = [queries]

    results = []
    top_k = config.get("top_k", 5)
    for query in queries:
        query_emb = query_embed_search(query, config["all_subqueries"], config["subquery_index"])
        query_emb = query_emb.reshape(1, -1)  
        distances, indices = config["faiss_index"].search(query_emb, top_k)
        results.extend([
            {
                "sub_query": query,
                "chunk_id": config["chunk_ids"][i],
                "passage": config["passages"][i],
                "score": float(distances[0][j])
            } for j, i in enumerate(indices[0])
        ])
    return results

def query_embed_search(query, all_queries_list, index):
    """
    Given a query, finds its embedding from the precomputed subqueries index.
    """
    try:
        position = all_queries_list.index(query)
        return index.reconstruct(position)
    except ValueError:
        raise ValueError(f"Query '{query}' not found in the list of all queries.")

def hybrid(query, config):

    # Check if the query is a list with at least two elements
    if isinstance(query, list) and len(query) >= 2:
        query_sparse = query[0]
        query_dense = query[1]
    else:
        print("Query should be a list of two queries: [sparse_query, dense_query].")

    intermediate_k = config.get("intermediate_k")
    final_k = config.get("final_k")
    mode = config.get("mode", "intersection")
    
    # Step 1: Sparse retrieval using query_sparse.
    sparse_config = config["sparse_config"].copy()
    sparse_config["top_k"] = intermediate_k
    sparse_results = config["sparse_func"](query_sparse, sparse_config)
    
    # Step 2: Dense retrieval using query_dense.
    dense_results = config["dense_func"](query_dense, config["dense_config"])
    
    # Step 3: Combine candidate sets.
    if mode == "intersection":
        candidate_ids = set(r["chunk_id"] for r in sparse_results)
        filtered_dense = [r for r in dense_results if r["chunk_id"] in candidate_ids]
        if not filtered_dense:
            filtered_dense = dense_results  # fallback if intersection is empty
    elif mode == "union":
        candidate_ids = set(r["chunk_id"] for r in sparse_results).union(
                        set(r["chunk_id"] for r in dense_results))
        dense_dict = {r["chunk_id"]: r for r in dense_results}
        filtered_dense = []
        for cid in candidate_ids:
            if cid in dense_dict:
                filtered_dense.append(dense_dict[cid])
            else:
                filtered_dense.append({"chunk_id": cid, "score": 0.0})
    else:
        raise ValueError("Invalid mode. Choose 'intersection' or 'union'.")
    
    # Step 4: Sort by dense score (descending) and select top final_k.
    sorted_dense = sorted(filtered_dense, key=lambda x: x["score"], reverse=True)
    #final_results = sorted_dense[:final_k]
    final_results = sorted_dense
    
    # ***** String Constraint Enforcement After Hybrid Retrieval *****
    # Ensure that query_dense is a string.
    query_dense = str(query_dense)
    # For each candidate, ensure the passage is a string.
    for candidate in final_results:
        candidate["passage"] = str(candidate.get("passage", ""))
    
    # Step 5 (Optional): Cross-Encoder re-ranking.
    if config.get("rerank") and "cross_encoder_model" in config:
        cross_encoder = config["cross_encoder_model"]
        # Before passing sentence pairs to cross_encoder.predict, ensure each passage is a string.
        sentence_pairs = []
        for candidate in final_results:
            passage = candidate.get("passage", "")
            if not isinstance(passage, str):
                passage = str(passage)
            sentence_pairs.append([query_dense, passage])
        try:
            cross_scores = cross_encoder.predict(sentence_pairs)
        except Exception as e:
            print("Error during cross encoder prediction:", e)
            cross_scores = [candidate["score"] for candidate in final_results]  # fallback
        for idx, candidate in enumerate(final_results):
            candidate["cross_encoder_score"] = float(cross_scores[idx])
        final_results = sorted(final_results, key=lambda x: x["cross_encoder_score"], reverse=True)
    
    return final_results[:final_k]

#########################################
# Step 1: Load Corpus & Embeddings
#########################################
def load_corpus(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    passages = [entry["passage"] for entry in data if "passage" in entry]
    chunk_ids = [entry["chunk_id"] for entry in data if "chunk_id" in entry]
    print(f"Loaded {len(passages)} corpus passages")
    return passages, chunk_ids

def retrieve_all_subqueries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    return [sq for item in qa_data for sq in item["sub_questions"]]

CORPUS_FILE = "../data/chunked_text_all_together_cleaned.json"
QA_PATH = "../data/QA_set"
QA_EMBEDDED_PATH = "../embedding/BAAI/bge-base-en-v1.5"
PM_PATH = "../performance"

passages, chunk_ids = load_corpus(CORPUS_FILE)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
doc_matrix = tfidf_vectorizer.fit_transform(passages)
tokenized_passages = [word_tokenize(p.lower()) for p in passages]
bm25 = BM25Okapi(tokenized_passages)
corpus_dense_index = faiss.read_index("../embedding/BAAI/bge-base-en-v1.5/hp_all_bge.index")

#########################################
# Step 2: Dataset + Retrieval Config Setup
#########################################
datasets = [
    {
        "name": "easy_single",
        "gt_path": os.path.join(QA_PATH, "easy_single_labeled.json"),
        "dense": {
            "subqueries": retrieve_all_subqueries(os.path.join(QA_PATH, "easy_single_labeled.json")),
            "subquery_index": faiss.read_index(os.path.join(QA_EMBEDDED_PATH, "easy_single_labeled_embeddings.index"))
        }
    },
    {
        "name": "medium_single",
        "gt_path": os.path.join(QA_PATH, "medium_single_labeled.json"),
        "dense": {
            "subqueries": retrieve_all_subqueries(os.path.join(QA_PATH, "medium_single_labeled.json")),
            "subquery_index": faiss.read_index(os.path.join(QA_EMBEDDED_PATH, "medium_single_labeled_embeddings.index"))
        }
    },
    {
        "name": "medium_multi",
        "gt_path": os.path.join(QA_PATH, "medium_multi_labeled.json"),
        "dense": {
            "subqueries": retrieve_all_subqueries(os.path.join(QA_PATH, "medium_multi_labeled.json")),
            "subquery_index": faiss.read_index(os.path.join(QA_EMBEDDED_PATH, "medium_multi_labeled_embeddings.index"))
        }
    },
    {
        "name": "hard_single",
        "gt_path": os.path.join(QA_PATH, "hard_single_labeled.json"),
        "dense": {
            "subqueries": retrieve_all_subqueries(os.path.join(QA_PATH, "hard_single_labeled.json")),
            "subquery_index": faiss.read_index(os.path.join(QA_EMBEDDED_PATH, "hard_single_labeled_embeddings.index"))
        }
    },
    {
        "name": "hard_multi",
        "gt_path": os.path.join(QA_PATH, "hard_multi_labeled.json"),
        "dense": {
            "subqueries": retrieve_all_subqueries(os.path.join(QA_PATH, "hard_multi_labeled.json")),
            "subquery_index": faiss.read_index(os.path.join(QA_EMBEDDED_PATH, "hard_multi_labeled_embeddings.index"))
        }
    }
]

retrieval_methods = {
    "tfidf": tfidf_retrieval,
    "bm25": bm25_retrieval,
    "dense": dense_retrieval_subqueries,
    "tfidf_dense": hybrid,
    "bm25_dense": hybrid,
    "dense_tfidf": hybrid,
    "dense_bm25": hybrid,
}

# Populate retrieval_config for each dataset
for ds in datasets:
    ds.setdefault("retrieval_config", {})
    ds["retrieval_config"]["tfidf"] = {
        "passages": passages,
        "chunk_ids": chunk_ids,
        "vectorizer": tfidf_vectorizer,
        "doc_matrix": doc_matrix,
        "top_k": 20,
        "query_type": "question"
    }
    ds["retrieval_config"]["bm25"] = {
        "passages": passages,
        "chunk_ids": chunk_ids,
        "bm25": bm25,
        "top_k": 20,
        "query_type": "question"
    }
    ds["retrieval_config"]["dense"] = {
        "passages": passages,
        "chunk_ids": chunk_ids,
        "all_subqueries": ds["dense"]["subqueries"],
        "subquery_index": ds["dense"]["subquery_index"],
        "faiss_index": corpus_dense_index,
        "top_k": 5,
        "query_type": "sub_questions"
    }
    for hybrid_method in ["tfidf_dense", "bm25_dense", "dense_tfidf", "dense_bm25"]:
        sparse_func = tfidf_retrieval if "tfidf" in hybrid_method else bm25_retrieval
        sparse_config = ds["retrieval_config"]["tfidf"] if "tfidf" in hybrid_method else ds["retrieval_config"]["bm25"]
        ds["retrieval_config"][hybrid_method] = {
            "sparse_func": sparse_func,
            "sparse_config": sparse_config,
            "dense_func": dense_retrieval_subqueries,
            "dense_config": ds["retrieval_config"]["dense"],
            "intermediate_k": 1000,
            "final_k": 20,
            "mode": "intersection",
            "query_type": "combine",
            "order": "sparse_dense" if hybrid_method.endswith("dense") else "dense_sparse"
        }

#########################################
# Step 3: Run Retrieval + Generation
#########################################
selected_method = "dense"
selected_dataset = "easy_single"
generator = CohereGenerator(model="command-r")

ds = next((d for d in datasets if d["name"] == selected_dataset), None)
with open(ds["gt_path"], "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)
print(f"Dataset '{ds['name']}': loaded {len(ground_truth_data)} ground truth examples.")

retrieval_func = retrieval_methods[selected_method]
retrieval_config = ds["retrieval_config"][selected_method]

test_results = []
predictions = []
references = []
questions = []

for example in ground_truth_data[:10]:
    origin_question = example["question"]
    subquestions = example["sub_questions"]
    query = [origin_question] + subquestions if len(subquestions) > 1 else [origin_question]
    retrieval_results = retrieval_func(query, retrieval_config)

    previous_qa, final_answer = generator.generation_answer(
        origin_question, subquestions, retrieval_results, top_k=5, max_tokens=60
    )

    predictions.append(final_answer)
    references.append(example["list of reference"])
    questions.append(origin_question)

    test_results.append({
        "question": origin_question,
        "sub_questions": subquestions,
        "retrieval_results": retrieval_results,
        "previous_qa": previous_qa,
        "final_answer": final_answer,
        "gt_answer": example['answer'],
        "references": example["list of reference"],
        "datasets": ds["name"]
    })

#########################################
# Step 4: Evaluation
#########################################
gen_metrics = MetricCollection({
    "sas": SASScore(),
    "GPT Score": CohereGPTScore(),
    "bertscore": BERTScore(model_name_or_path="bert-base-uncased", batch_size=16)
})

gen_metrics.update(predictions, [r["gt_answer"] for r in test_results], questions, metric_type="generation")
gen_results = gen_metrics.compute(metric_type="generation")

print(f"\nGeneration evaluation metrics for dataset {ds['name']}:")
print(json.dumps(gen_results, indent=2))