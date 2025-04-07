import json
import os
import sys
from dotenv import load_dotenv

# Add the parent directory (NLP_project) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import itertools
import faiss
import sentence_transformers
from generation.cohere_generation import CohereGenerator
from retrieval import load_passages_and_chunk_ids
import transformers


def process_question_block(block_type, sub_questions_key, item, model, generator, passages, chunk_ids, all_chunks_index):
    results = []
    sub_questions = item[sub_questions_key]
    perms = list(itertools.permutations(sub_questions))

    # Pre-encode all sub-questions
    subq_to_embedding = {
        sq: model.encode(sq, convert_to_numpy=True, normalize_embeddings=True)
        for sq in set(sub_questions)
    }

    for perm in perms:
        question_list = list(perm)
        retrieve_results = []
        for query in question_list:
            query_emb = subq_to_embedding[query]
            distances, indices = all_chunks_index.search(query_emb.reshape(1, -1), 5)
            retrieve_results.extend([
                {
                    "sub_query": query,
                    "chunk_id": chunk_ids[i],
                    "passage": passages[i],
                    "score": float(distances[0][j])
                } for j, i in enumerate(indices[0])
            ])
            # print(retrieve_results)
        _, final_answer = generator.generation_answer(item['question'], question_list, retrieve_results, top_k=5)
        results.append({
            "graphy type": block_type,
            "question": item["question"],
            "sub_questions_ordered": question_list,
            "final_answers": final_answer
        })

    return results


if __name__ == "__main__":
    QA_Path = 'data/QA_set'
    qo_test_file_path = "data/different_subquestions_from_test_set.json"
    with open(qo_test_file_path, "r") as f:
        qo_test_data = json.load(f)

    load_dotenv("key.env")
    api_key = os.environ.get('COHERE_API_KEY')
    generator = CohereGenerator(api_key=api_key)
    model = sentence_transformers.SentenceTransformer('CatkinChen/BAAI_bge-base-en-v1.5_retrieval_finetuned_v1')
    passages, chunk_ids = load_passages_and_chunk_ids()
    all_chunks_index = faiss.read_index('embedding/BAAI/bge-base-en-v1.5_finetuned/hp_all_BAAI/bge-base-en-v1.5_finetuned.index')

    final_answers_list = []

    for item in qo_test_data:
        final_answers_list += process_question_block("v structure", "V sub_questions", item, model, generator, passages, chunk_ids, all_chunks_index)
        final_answers_list += process_question_block("chain", "Chain sub_questions", item, model, generator, passages, chunk_ids, all_chunks_index)
    
    with open("./QO_experiment_results.json", "w") as f:
        json.dump(final_answers_list, f)