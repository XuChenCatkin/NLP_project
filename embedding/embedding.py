import json
import gc
from pathlib import Path
from typing import Callable, List, Dict, Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    PreTrainedModel, PreTrainedTokenizer
)
import os
# --- Configuration ---
# Use Pathlib for cleaner path management
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
EMBEDDING_PATH = BASE_DIR / "embedding"
ALL_CHUNKS_FILE = DATA_PATH / "chunked_text_all_together_cleaned.json"
QA_PATH = DATA_PATH / "QA"

# Determine processing device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Loading ---

def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a list of dictionaries from a JSON file."""
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_passages(data: List[Dict[str, Any]]) -> List[str]:
    """Extracts passages from loaded data."""
    return [entry["passage"] for entry in data if "passage" in entry]

def get_questions(data: List[Dict[str, Any]]) -> List[str]:
    """Extracts questions from loaded data."""
    return [item["question"] for item in data if "question" in item]

def get_subquestions(data: List[Dict[str, Any]]) -> List[str]:
    """Extracts all sub-questions from loaded data."""
    sub_questions = []
    for item in data:
        sub_questions.extend(item.get("sub_questions", []))
    return sub_questions

# --- Embedding Logic ---

def embed_with_sentencetransformer(texts: List[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Generates normalized embeddings using a SentenceTransformer model."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

def embed_with_dpr(texts: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int = 32) -> np.ndarray:
    """Generates normalized embeddings using a DPR-style model."""
    model.eval()
    all_embeddings = []
    
    # Use DataLoader for efficient batching
    data_loader = DataLoader(texts, batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding DPR"):
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            embeddings = model(**inputs).pooler_output
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

# --- FAISS Indexing ---

def store_faiss_index(embeddings: np.ndarray, index_path: Path):
    """Creates and stores a FAISS index from pre-normalized embeddings."""
    if embeddings.size == 0:
        print(f"Warning: No embeddings to store for {index_path.name}. Skipping.")
        return

    dimension = embeddings.shape[1]
    # Use IndexFlatIP for cosine similarity on normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Ensure parent directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS index with {index.ntotal} vectors stored at: {index_path}")

# --- Main Processing Pipeline ---

def run_processing_pipeline(
    model_name: str,
    embedder_func: Callable[[List[str]], np.ndarray],
    data_source: Path,
    text_extractor: Callable[[List[Dict[str, Any]]], List[str]],
    output_filename: str
):
    """Orchestrates loading data, embedding, and storing the index."""
    print("-" * 50)
    print(f"Starting processing for '{model_name}' on '{data_source.name}'")
    
    # 1. Load Data
    raw_data = load_json_data(data_source)
    if not raw_data:
        return # Exit if no data
        
    # 2. Extract relevant texts
    texts_to_embed = text_extractor(raw_data)
    print(f"Found {len(texts_to_embed)} items to embed.")
    if not texts_to_embed:
        return

    # 3. Generate Embeddings
    embeddings = embedder_func(texts_to_embed)

    # 4. Store FAISS Index
    index_path = EMBEDDING_PATH / model_name / output_filename
    store_faiss_index(embeddings, index_path)

    # Clean up GPU memory
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    print("-" * 50)

# --- Execution ---

def main():
    """Main function to run the embedding and indexing workflows."""
    
    # --- BGE Model Workflow ---
    print("\nInitializing BGE model...")
    bge_model_name = "bge-base-finetuned"
    bge_model = SentenceTransformer("CatkinChen/BAAI_bge-base-en-v1.5_retrieval_finetuned_v1", device=DEVICE)
    bge_embedder = lambda texts: embed_with_sentencetransformer(texts, bge_model)

    # Process all passages with BGE
    run_processing_pipeline(
        model_name=bge_model_name,
        embedder_func=bge_embedder,
        data_source=ALL_CHUNKS_FILE,
        text_extractor=get_passages,
        output_filename=f"{bge_model_name}_passages.index"
    )

    # Process QA files for questions and subquestions with BGE
    qa_files = list(QA_PATH.glob("*_labeled.json"))
    for qa_file in qa_files:
        base_name = qa_file.stem
        # Process original questions
        run_processing_pipeline(
            model_name=bge_model_name,
            embedder_func=bge_embedder,
            data_source=qa_file,
            text_extractor=get_questions,
            output_filename=f"{base_name}_questions.index"
        )
        # Process sub-questions
        run_processing_pipeline(
            model_name=bge_model_name,
            embedder_func=bge_embedder,
            data_source=qa_file,
            text_extractor=get_subquestions,
            output_filename=f"{base_name}_subquestions.index"
        )
        
    # # --- DPR Model Workflow ---
    # print("\nInitializing DPR models...")
    # dpr_model_name = "dpr"
    
    # # Context model for encoding passages
    # ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    # dpr_context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(DEVICE)
    # dpr_ctx_embedder = lambda texts: embed_with_dpr(texts, dpr_context_model, ctx_tokenizer)
    
    # # Question model for encoding queries
    # qs_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    # dpr_question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(DEVICE)
    # dpr_qs_embedder = lambda texts: embed_with_dpr(texts, dpr_question_model, qs_tokenizer)
    
    # # Process all passages with DPR Context Encoder
    # run_processing_pipeline(
    #     model_name=dpr_model_name,
    #     embedder_func=dpr_ctx_embedder,
    #     data_source=ALL_CHUNKS_FILE,
    #     text_extractor=get_passages,
    #     output_filename=f"{dpr_model_name}_passages.index"
    # )

    # # Process QA files with DPR Question Encoder
    # for qa_file in qa_files:
    #     base_name = qa_file.stem
    #     # Process original questions
    #     run_processing_pipeline(
    #         model_name=dpr_model_name,
    #         embedder_func=dpr_qs_embedder,
    #         data_source=qa_file,
    #         text_extractor=get_questions,
    #         output_filename=f"{base_name}_questions.index"
    #     )
    #     # Process sub-questions
    #     run_processing_pipeline(
    #         model_name=dpr_model_name,
    #         embedder_func=dpr_qs_embedder,
    #         data_source=qa_file,
    #         text_extractor=get_subquestions,
    #         output_filename=f"{base_name}_subquestions.index"
    #     )

if __name__ == "__main__":
    main()