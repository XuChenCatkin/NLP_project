# src/data_processing.py
import os
import json

def load_corpus(filepath: str) -> tuple[list[dict], list[str], list[str]]:
    """Loads the entire corpus, returning the raw data, passages, and chunk IDs."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing corpus file: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
        
    passages = [entry["passage"] for entry in corpus_data]
    chunk_ids = [entry["chunk_id"] for entry in corpus_data]
    
    print(f"Loaded {len(passages)} passages from corpus.")
    return corpus_data, passages, chunk_ids

def load_queries_from_qa(filepath: str) -> list[str]:
    """Extracts all sub-questions from a QA dataset file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    subquestion_list = [
        sub for item in qa_data for sub in item.get('sub_questions', [])
    ]
    print(f"Loaded {len(subquestion_list)} sub-questions.")
    return subquestion_list