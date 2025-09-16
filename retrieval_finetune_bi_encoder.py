from __future__ import annotations

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer, InputExample, models, util
from sentence_transformers.datasets import NoDuplicatesDataLoader  # not used (custom collate)
from sentence_transformers.evaluation import InformationRetrievalEvaluator  # not used (custom eval)
from sentence_transformers.util import cos_sim

from sklearn.model_selection import train_test_split
from huggingface_hub import notebook_login
import wandb
import logging
from utils import create_logger

import time
import faiss

# Optional (scheduler)
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------
notebook_login()

DATA_PATH = "./data"
QA_PATH = f"{DATA_PATH}/QA_set"
QA_EMBEDDED_PATH = f"{DATA_PATH}/QA_set_embedded"
EMBEDDING_PATH = f"./embedding"

EASY = f"{QA_PATH}/easy_single_labeled.json"
MEDIUM_S = f"{QA_PATH}/medium_single_labeled.json"
MEDIUM_M = f"{QA_PATH}/medium_multi_labeled.json"
HARD_S = f"{QA_PATH}/hard_single_labeled.json"
HARD_M = f"{QA_PATH}/hard_multi_labeled.json"

CORPUS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"

MODEL = "dpr"

# Load prebuilt FAISS + subqueries (as in your original code)
from retrieval import dense_retrieval_subqueries_for_finetune, retrieve_all_subqueries  # noqa

subqueries_easy = retrieve_all_subqueries(EASY)
subqueries_medium_s = retrieve_all_subqueries(MEDIUM_S)
subqueries_medium_m = retrieve_all_subqueries(MEDIUM_M)
subqueries_hard_s = retrieve_all_subqueries(HARD_S)
subqueries_hard_m = retrieve_all_subqueries(HARD_M)

index_easy = faiss.read_index(f"{EMBEDDING_PATH}/{MODEL}/easy_single_labeled_embeddings.index")
index_medium_s = faiss.read_index(f"{EMBEDDING_PATH}/{MODEL}/medium_single_labeled_embeddings.index")
index_medium_m = faiss.read_index(f"{EMBEDDING_PATH}/{MODEL}/medium_multi_labeled_embeddings.index")
index_hard_s = faiss.read_index(f"{EMBEDDING_PATH}/{MODEL}/hard_single_labeled_embeddings.index")
index_hard_m = faiss.read_index(f"{EMBEDDING_PATH}/{MODEL}/hard_multi_labeled_embeddings.index")

corpus_index = faiss.read_index(f'./embedding/{MODEL}/hp_all_{MODEL}.index')
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    corpus = json.load(f)

def load_index_and_all_subqueries(category):
    if category == "easy_single_labeled":
        return index_easy, subqueries_easy
    elif category == "medium_single_labeled":
        return index_medium_s, subqueries_medium_s
    elif category == "medium_multi_labeled":
        return index_medium_m, subqueries_medium_m
    elif category == "hard_single_labeled":
        return index_hard_s, subqueries_hard_s
    elif category == "hard_multi_labeled":
        return index_hard_m, subqueries_hard_m
    else:
        raise ValueError("Unknown category")

# -----------------------------
# ARGUMENTS
# -----------------------------
args = {
    "model_name_query": "BAAI/bge-base-en-v1.5",       # query tower init
    "model_name_doc":   "BAAI/bge-base-en-v1.5",       # doc tower init
    "corpus_file": CORPUS_FILE,
    "easy_file": EASY,
    "medium_single_file": MEDIUM_S,
    "medium_multi_file": MEDIUM_M,
    "hard_single_file": HARD_S,
    "hard_multi_file": HARD_M,
    "batch_size": 2,
    "huggingfaceusername": "CatkinChen",
    "wandbusername": "xiangzhang350-ucl",
    "epochs": 30,
    "margin": 0.3,              # not used by MNRL
    "test_size": 0.2,
    "random_state": 42,
    "top_k": 5,
    "temperature": 20.0,        # MNRL temperature (logit scale)
    "max_len_query": 128,
    "max_len_doc": 256,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
}

def parse_args():
    this_args = args.copy()
    this_args['project'] = f"dualbert-mnrl-finetune"
    this_args['experiment_name'] = f"dualbert-mnrl-run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    this_args["repo_id_query"] = f"{this_args['huggingfaceusername']}/dualbert_query_{ts}"
    this_args["repo_id_doc"]   = f"{this_args['huggingfaceusername']}/dualbert_doc_{ts}"
    simple_name_args = SimpleNamespace(**this_args)
    return simple_name_args

# -----------------------------
# DATA HELPERS
# -----------------------------
def load_json_data(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: File {path} does not contain a list. Skipping.")
                return []
            return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

def random_sample(data, n):
    if len(data) < n:
        return data
    return np.random.choice(data, n, replace=False).tolist()

def hard_negative_mining(item):
    try:
        references = item["list of reference"]
        reference_ids = set([ref["ref_id"] for ref in references]) if references else set()
        _sub_questions = item.get("sub_questions", [])
        index_file, all_subquestion_list = load_index_and_all_subqueries(item["category"])

        # Placeholder: you can call dense_retrieval_subqueries_for_finetune(...) to get better negatives
        rand_neg_list = random_sample(
            [corpus[i] for i in range(len(corpus)) if (corpus[i].get('chunk_id') not in reference_ids)],
            5
        )
        negative_retrieval_set = set()
        for negative_retrieval in rand_neg_list:
            cid = negative_retrieval.get('chunk_id')
            if cid is not None and cid not in reference_ids:
                negative_retrieval_set.add(cid)

        negative_retrieval_list = list(negative_retrieval_set)
        negative_retrievals = [corpus[int(neg_id) - 1] for neg_id in negative_retrieval_list]
        assert len(negative_retrievals) >= 1, f"Not enough negative samples found for item: {item}"
        return negative_retrievals
    except KeyError:
        raise (f"KeyError: 'list of reference' not found in item: {item}")
    except Exception as e:
        raise (f"Unexpected error: {e}")

def process_data_MNRL(data):
    """
    Returns InputExample with texts = [query, positive, neg1, neg2, ...]
    """
    examples = []
    query_map = {}
    relevant_map = {}
    for item in data:
        question = item["question"]
        refs = item["list of reference"]
        ref_len = len(refs)

        query_id = item['category'] + '_' + str(item['id'])
        query_map[query_id] = question
        relevant_map[query_id] = set([str(ref['ref_id']) for ref in refs])

        negative_list = hard_negative_mining(item)

        for i in range(ref_len if ref_len > 0 else 1):
            if ref_len == 0:
                positive_enhanced = ''
            else:
                passage_text = refs[i]['passage']
                positive_enhanced = f"{passage_text}"
            negative_enhanced = []
            for j in range(len(negative_list)):
                neg_passage_text = negative_list[j]['passage']
                negative_enhanced.append(f"{neg_passage_text}")
            examples.append(InputExample(texts=[question, positive_enhanced] + negative_enhanced))
    return examples, query_map, relevant_map

# -----------------------------
# DUAL-TOWER LOSS & COLLATE
# -----------------------------
class TwoTowerMNRL(nn.Module):
    """
    Multiple Negatives Ranking Loss with separate encoders for queries and documents.
    Supports batches shaped by `dual_mnrl_collate`.
    """
    def __init__(self,
                 query_encoder: SentenceTransformer,
                 doc_encoder: SentenceTransformer,
                 scale: float = 20.0,
                 similarity_fct=util.cos_sim):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.ce = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels=None):
        # sentence_features = [query_features, candidate_features]
        q_emb = self.query_encoder(sentence_features[0])["sentence_embedding"]           # (B, d)
        c_emb = self.doc_encoder(sentence_features[1])["sentence_embedding"]             # (B + K, d)

        B = q_emb.size(0)
        # scores against all candidates (first B are positives, rest are negatives)
        scores = self.similarity_fct(q_emb, c_emb) * self.scale                          # (B, B+K)
        targets = torch.arange(B, device=scores.device)                                   # positives aligned 0..B-1
        return self.ce(scores, targets)

def dual_mnrl_collate(examples: List[InputExample]):
    """
    Produces:
      [query_features, candidate_features]
    where candidate_features = [pos_1..pos_B, negs...]
    """
    queries = [ex.texts[0] for ex in examples]
    positives = [ex.texts[1] for ex in examples]
    negatives = []
    for ex in examples:
        if len(ex.texts) > 2:
            negatives.extend(ex.texts[2:])
    candidates = positives + negatives

    # Use global encoders defined in train(); we will bind tokenize lambdas after creation
    q_features = dual_mnrl_collate.query_tokenizer(queries)
    c_features = dual_mnrl_collate.doc_tokenizer(candidates)
    return [q_features, c_features], None

# These two callables will be injected at runtime:
dual_mnrl_collate.query_tokenizer = None
dual_mnrl_collate.doc_tokenizer = None

# -----------------------------
# SIMPLE IR EVAL (dual towers)
# -----------------------------
@torch.no_grad()
def evaluate_ir(query_encoder: SentenceTransformer,
                doc_encoder: SentenceTransformer,
                query_map: Dict[str, str],
                corpus_map: Dict[str, str],
                relevant_map: Dict[str, set],
                ks=(1, 5, 10)) -> Dict[str, Any]:
    """
    Computes Recall@K (micro-averaged over queries).
    """
    queries = [query_map[qid] for qid in query_map.keys()]
    qids = list(query_map.keys())

    corpus_ids = list(corpus_map.keys())
    corpus_texts = [corpus_map[cid] for cid in corpus_ids]

    q_emb = query_encoder.encode(queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    d_emb = doc_encoder.encode(corpus_texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)

    sims = cos_sim(q_emb, d_emb)  # (num_queries, num_docs)

    recalls = {f"R@{k}": 0 for k in ks}
    for i, qid in enumerate(qids):
        rel_ids = relevant_map[qid]
        top_vals, top_idx = torch.topk(sims[i], k=max(ks))
        top_cids = [corpus_ids[j] for j in top_idx.tolist()]
        for k in ks:
            hit = any(cid in rel_ids for cid in top_cids[:k])
            if hit:
                recalls[f"R@{k}"] += 1

    N = len(qids)
    for k in ks:
        recalls[f"R@{k}"] = recalls[f"R@{k}"] / max(1, N)

    return recalls

# -----------------------------
# UTILS
# -----------------------------
def batch_to_device(features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in features.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

@torch.no_grad()
def compute_pos_neg_similarity(query_encoder: SentenceTransformer,
                               doc_encoder: SentenceTransformer,
                               features_batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]) -> Tuple[float, float]:
    """
    For logging: average positive similarity (diagonal) and average 'negative' similarity
    across the batch (off-diagonal inside the first B columns).
    """
    q_emb = query_encoder(features_batch[0])['sentence_embedding']
    c_emb = doc_encoder(features_batch[1])['sentence_embedding']
    B = q_emb.size(0)
    sims = cos_sim(q_emb, c_emb)
    pos_sim = (sims[:, :B].diag()).mean().item()
    off_diag = sims[:, :B]
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=off_diag.device)
    neg_sim = off_diag[neg_mask].mean().item()
    return pos_sim, neg_sim

# -----------------------------
# TRAIN (Custom loop updating BOTH towers)
# -----------------------------
def train(run_args, logger: logging.Logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build two towers
    logger.info(f"Loading query model {run_args.model_name_query} on {device}")
    q_tr = models.Transformer(run_args.model_name_query, max_seq_length=run_args.max_len_query)
    q_pool = models.Pooling(q_tr.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    query_encoder = SentenceTransformer(modules=[q_tr, q_pool], device=device)

    logger.info(f"Loading doc model {run_args.model_name_doc} on {device}")
    d_tr = models.Transformer(run_args.model_name_doc, max_seq_length=run_args.max_len_doc)
    d_pool = models.Pooling(d_tr.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    doc_encoder = SentenceTransformer(modules=[d_tr, d_pool], device=device)

    # Tokenizer hooks for collate
    dual_mnrl_collate.query_tokenizer = query_encoder.tokenize
    dual_mnrl_collate.doc_tokenizer = doc_encoder.tokenize

    # Load data
    logger.info("Loading data")
    corpus_items = load_json_data(run_args.corpus_file)
    corpus_map = dict([(str(item["chunk_id"]), item['passage']) for item in corpus_items])

    easy = load_json_data(run_args.easy_file)
    medium_single = load_json_data(run_args.medium_single_file)
    medium_multi = load_json_data(run_args.medium_multi_file)
    hard_single = load_json_data(run_args.hard_single_file)
    hard_multi = load_json_data(run_args.hard_multi_file)

    train_data_easy, test_data_easy = train_test_split(easy, test_size=run_args.test_size, random_state=run_args.random_state)
    train_data_medium_single, test_data_medium_single = train_test_split(medium_single, test_size=run_args.test_size, random_state=run_args.random_state)
    train_data_medium_multi, test_data_medium_multi = train_test_split(medium_multi, test_size=run_args.test_size, random_state=run_args.random_state)
    train_data_hard_single, test_data_hard_single = train_test_split(hard_single, test_size=run_args.test_size, random_state=run_args.random_state)
    train_data_hard_multi, test_data_hard_multi = train_test_split(hard_multi, test_size=run_args.test_size, random_state=run_args.random_state)

    train_data = (train_data_easy + train_data_medium_single + train_data_medium_multi + train_data_hard_single + train_data_hard_multi)
    test_data = (test_data_easy + test_data_medium_single + test_data_medium_multi + test_data_hard_single + test_data_hard_multi)

    # Save splits
    with open(f"data/finetune_train_data_test_size_{run_args.test_size}_random_state_{run_args.random_state}.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    with open(f"data/finetune_test_data_test_size_{run_args.test_size}_random_state_{run_args.random_state}.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    logger.info(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")

    # Build InputExamples (query, pos, negs...)
    train_examples, train_query_map, train_relevant_map = process_data_MNRL(train_data)
    test_examples,  test_query_map,  test_relevant_map  = process_data_MNRL(test_data)

    # Persist processed for debugging
    def to_dict_list(examples):
        return [{"anchor": ex.texts[0], "positive": ex.texts[1], "negative": ex.texts[2:]} for ex in examples]
    with open (f"data/finetune_train_triplets_test_size_{run_args.test_size}_random_state_{run_args.random_state}_processed.json", "w", encoding="utf-8") as f:
        json.dump(to_dict_list(train_examples), f, indent=4)

    # DataLoader with custom collate
    train_dataloader = DataLoader(
        train_examples,
        batch_size=run_args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dual_mnrl_collate
    )

    # Loss (holds both towers)
    train_loss = TwoTowerMNRL(query_encoder, doc_encoder, scale=run_args.temperature)

    # Log config
    wandb.config.update({
        "epochs": run_args.epochs,
        "batch_size": run_args.batch_size,
        "query_model_name": run_args.model_name_query,
        "doc_model_name": run_args.model_name_doc,
        "train_data_size": len(train_examples),
        "test_data_size": len(test_examples),
        "temperature": run_args.temperature,
        "lr": run_args.lr,
        "weight_decay": run_args.weight_decay,
        "warmup_ratio": run_args.warmup_ratio,
        "grad_clip": run_args.grad_clip,
    })

    # ====== OPTIMIZER + SCHEDULER (both towers) ======
    params = [
        {"params": [p for p in query_encoder.parameters() if p.requires_grad]},
        {"params": [p for p in doc_encoder.parameters() if p.requires_grad]},
    ]
    optimizer = AdamW(params, lr=run_args.lr, weight_decay=run_args.weight_decay)

    total_steps = len(train_dataloader) * run_args.epochs
    warmup_steps = int(run_args.warmup_ratio * total_steps) if total_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps) if total_steps > 0 else None

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Train
    logger.info("Training model (dual-tower MNRL) with custom loop")
    query_encoder.train()
    doc_encoder.train()

    global_step = 0
    for epoch in range(run_args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (features, _labels) in enumerate(train_dataloader):
            q_features = batch_to_device(features[0], device)
            c_features = batch_to_device(features[1], device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = train_loss([q_features, c_features])  # CE loss

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            if run_args.grad_clip and run_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), run_args.grad_clip)
                torch.nn.utils.clip_grad_norm_(doc_encoder.parameters(), run_args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                # quick similarity diagnostics on the *current* batch
                with torch.no_grad():
                    pos_sim, neg_sim = compute_pos_neg_similarity(query_encoder, doc_encoder, (q_features, c_features))
                wandb.log({
                    "train_loss": float(loss.detach().cpu()),
                    "epoch": epoch,
                    "train_step": global_step,
                    "avg_positive_similarity_batch": pos_sim,
                    "avg_negative_similarity_batch": neg_sim,
                })

        epoch_time = time.time() - t0
        with torch.no_grad():
            eval_scores = evaluate_ir(query_encoder, doc_encoder, test_query_map, corpus_map, test_relevant_map, ks=(1,5,10))
        # wandb.log({ "epoch": epoch})
        wandb.log({
                   **{f"eval_{k}": v for k, v in eval_scores.items()},
                   "epoch_avg_loss": epoch_loss / max(1, len(train_dataloader)),
                   "epoch_time_sec": epoch_time,
                   "epoch": epoch})

    # Switch to eval before IR eval
    query_encoder.eval()
    doc_encoder.eval()

    logger.info("Evaluating model (IR Recall@K)")
    # eval_scores = evaluate_ir(query_encoder, doc_encoder, test_query_map, corpus_map, test_relevant_map, ks=(1,5,10))
    # wandb.log({**{f"eval_{k}": v for k, v in eval_scores.items()}, "epoch": run_args.epochs})

    wandb.finish()

    # Save & push each tower
    logger.info("Saving models locally")
    query_encoder.save("rag-dual/query-encoder")
    doc_encoder.save("rag-dual/passage-encoder")

    logger.info("Pushing models to Hugging Face Hub")
    try:
        query_encoder.push_to_hub(repo_id=run_args.repo_id_query)
        doc_encoder.push_to_hub(repo_id=run_args.repo_id_doc)
    except Exception as e:
        logger.warning(f"Push to hub failed: {e}. Models are saved locally at rag-dual/.")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    run_args = parse_args()
    wandb.init(project=run_args.project, name=run_args.experiment_name, entity=run_args.wandbusername)
    logger = create_logger(run_args.experiment_name, console_output=True)
    logger.info(f"Using arguments: {run_args}")
    train(run_args, logger)
