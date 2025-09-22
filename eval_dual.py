import json
import torch
from tqdm import tqdm
from typing import Dict, Any, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# CONFIGURATION
# -----------------------------
# --- Paths to your trained models ---
MODEL_QUERY_PATH = "rag-dual/query-encoder"
MODEL_DOC_PATH = "rag-dual/passage-encoder"

# --- Paths to your data files ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CORPUS_FILE_PATH = "./data/chunked_text_all_together_cleaned.json"
TEST_DATA_PATH = f"data/finetune_test_data_test_size_{TEST_SIZE}_random_state_{RANDOM_STATE}.json"

# --- Evaluation Parameters ---
EVAL_BATCH_SIZE = 64
RECALL_KS = (1, 3, 5, 10)


# ------------------------------------
# EVALUATION FUNCTION (Slightly modified for clarity in loops)
# ------------------------------------
@torch.no_grad()
def evaluate_ir(
    query_encoder: SentenceTransformer,
    doc_encoder: SentenceTransformer,
    corpus_embeddings: torch.Tensor,
    corpus_ids: list,
    query_map: Dict[str, str],
    relevant_map: Dict[str, Set[str]],
    ks: tuple = (1, 5, 10),
    category_name: str = "all",
) -> Dict[str, Any]:
    """
    Computes Recall@K and MRR for a given set of queries.
    This version accepts pre-computed corpus embeddings to avoid re-encoding in a loop.
    """
    queries = list(query_map.values())
    qids = list(query_map.keys())

    q_emb = query_encoder.encode(
        queries,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=EVAL_BATCH_SIZE,
        show_progress_bar=True,
        desc=f"Encoding queries ({category_name})"
    )

    # Use pre-computed corpus embeddings
    sims = util.cos_sim(q_emb, corpus_embeddings)

    recalls = {f"Recall@{k}": 0.0 for k in ks}
    mrr_sum = 0.0

    for i in tqdm(range(len(qids)), desc=f"Evaluating metrics ({category_name})"):
        qid = qids[i]
        rel_ids = relevant_map.get(qid, set())
        if not rel_ids:
            continue

        top_vals, top_idx = torch.topk(sims[i], k=max(ks))
        top_cids = [corpus_ids[j] for j in top_idx.tolist()]

        for k in ks:
            if any(cid in rel_ids for cid in top_cids[:k]):
                recalls[f"Recall@{k}"] += 1

        for rank, cid in enumerate(top_cids):
            if cid in rel_ids:
                mrr_sum += 1 / (rank + 1)
                break

    N = len(qids)
    for k in ks:
        recalls[f"Recall@{k}"] = recalls[f"Recall@{k}"] / N if N > 0 else 0
    
    mrr = mrr_sum / N if N > 0 else 0
    
    results = recalls
    results["MRR"] = mrr
    results["Query_Count"] = N
    
    return results

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}\n")

    # 1. Load the fine-tuned models
    print(f"Loading query encoder from: {MODEL_QUERY_PATH}")
    query_encoder = SentenceTransformer(MODEL_QUERY_PATH, device=device)
    print(f"Loading passage encoder from: {MODEL_DOC_PATH}")
    doc_encoder = SentenceTransformer(MODEL_DOC_PATH, device=device)
    query_encoder.eval()
    doc_encoder.eval()

    # 2. Load the corpus and test data
    print(f"\nLoading corpus from: {CORPUS_FILE_PATH}")
    with open(CORPUS_FILE_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    print(f"Loading test data from: {TEST_DATA_PATH}")
    try:
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at '{TEST_DATA_PATH}'. Please check configuration.")
        exit()

    # 3. Pre-encode the entire corpus ONCE to save time
    corpus_map = {str(item['chunk_id']): item['passage'] for item in corpus_data}
    corpus_ids = list(corpus_map.keys())
    corpus_texts = list(corpus_map.values())
    
    print(f"\nPre-encoding {len(corpus_texts)} corpus documents... (This may take a moment)")
    corpus_embeddings = doc_encoder.encode(
        corpus_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=EVAL_BATCH_SIZE,
        show_progress_bar=True,
        desc="Encoding corpus"
    )
    print("Corpus encoding complete.")

    # 4. Group test data by category
    categorized_test_data = defaultdict(list)
    for item in test_data:
        categorized_test_data[item['category']].append(item)
    
    # Also create an "overall" category to evaluate all test data together
    categorized_test_data['overall'] = test_data
    
    all_results = {}

    # 5. Loop through each category and evaluate
    for category, items in sorted(categorized_test_data.items()):
        print(f"\n{'='*20} Evaluating Category: {category} {'='*20}")
        
        # Create maps for this specific category
        query_map_cat = {}
        relevant_map_cat = {}
        for item in items:
            query_id = f"{item['category']}_{item['id']}"
            query_map_cat[query_id] = item['question']
            relevant_map_cat[query_id] = {str(ref['ref_id']) for ref in item.get('list of reference', [])}

        if not query_map_cat:
            print(f"No queries found for category '{category}'. Skipping.")
            continue

        # Run evaluation
        eval_results = evaluate_ir(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            corpus_embeddings=corpus_embeddings,
            corpus_ids=corpus_ids,
            query_map=query_map_cat,
            relevant_map=relevant_map_cat,
            ks=RECALL_KS,
            category_name=category
        )
        all_results[category] = eval_results

    # 6. Print the final summary report
    print(f"\n\n{'='*25} 📊 Final Evaluation Summary {'='*25}")
    
    # Define the order for printing results
    category_order = sorted([cat for cat in all_results.keys() if cat != 'overall'])
    category_order.append('overall')

    for category in category_order:
        results = all_results[category]
        query_count = results.pop("Query_Count")
        print(f"\n--- Category: {category} ({query_count} queries) ---")
        for metric, score in results.items():
            print(f"{metric:<12}: {score:.4f}")
    print(f"\n{'='*73}")