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


def dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index, all_chucks, top_k=5,params = faiss.SearchParameters()):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    for query in queries:
        #query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        query_emb = query_embed_search(query, all_queries_list, sub_queries_index)
        query_emb = query_emb.reshape(1, -1)  # Reshapes to (1, d)
        distances, indices = faiss_index.search(query_emb, top_k,params=params)
        results.extend([ all_chucks[i] for _, i in enumerate(indices[0]) ])
    return results

# def KG_dense_retrieval(queries, all_queries_list, sub_queries_index, faiss_index_kg, faiss_index_chunk, all_chucks, relation_to_kgid_map, top_k=5):
#     if isinstance(queries, str):
#         queries = [queries]

#     results = []
#     kg_ids = []
#     for query in queries:
#         #query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
#         query_emb = query_embed_search(query, all_queries_list, sub_queries_index)
#         query_emb = query_emb.reshape(1, -1)  # Reshapes to (1, d)
#         distances, indices = faiss_index_kg.search(query_emb,1000)
#         entities = process_gpt(query)
#         shared_kg = find_chunk_id(entities)
#         print(shared_kg)
#         relation_rank_index = indices[0]
#         print(relation_rank_index)
#         for i in relation_rank_index:
#             if relation_to_kgid_map[i] in shared_kg:
#                 kg_ids.append(relation_to_kgid_map[i])
#         if len(kg_ids) == 0:
#             print('KG method failed')
#             print('return the dense retrieval')
#             return dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index_chunk, all_chucks, top_k=5)
#         else:
#             chunk_id_list = []
#             for kg_id in kg_ids:
#                 chunk_id_list.extend([i for i in range(5*kg_id, 5*kg_id+5)])
#                 chunk_id_list = list(set(chunk_id_list))
#             sel = faiss.IDSelectorArray(chunk_id_list)
#             params = faiss.SearchParameters()
#             params.sel = sel
#             return dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index_chunk, all_chucks, top_k=5,params=params)

def KG_dense_retrieval(queries, all_queries_list, sub_queries_index, faiss_index_kg, faiss_index_chunk, all_chucks, relation_to_kgid_map, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []
    kg_ids = []
    
    for query in queries:
        entities = process_gpt(query)
        shared_kg = find_chunk_id(entities)
        kg_ids.extend(shared_kg)
    kg_ids = list(set(kg_ids))


    relation_list = []
    for i in range(len(relation_to_kgid_map)):
        if relation_to_kgid_map[i] in kg_ids:
            # print(i+1)
            relation_list.append(i+1)

    kg_ids_1 = []
    for query in queries:
        #query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        sel_kg = faiss.IDSelectorArray(relation_list)
        # print(sel_kg)
        params = faiss.SearchParameters()
        params.sel = sel_kg
        
        query_emb = query_embed_search(query, all_queries_list, sub_queries_index)
        query_emb = query_emb.reshape(1, -1)  # Reshapes to (1, d)

        distances, indices = faiss_index_kg.search(query_emb,100,params=params)
        relation_rank_index = indices[0]
        for i in relation_rank_index:
            if relation_to_kgid_map[i] in kg_ids:
                kg_ids_1.append(relation_to_kgid_map[i])
    if len(kg_ids_1) == 0:
        print('KG method failed')
        print('return the dense retrieval')
        return dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index_chunk, all_chucks, top_k=5)
    else:
        chunk_id_list = []
        for kg_id in kg_ids_1:
            chunk_id_list.extend([i for i in range(5*kg_id-4, 5*kg_id+1)])
            chunk_id_list = list(set(chunk_id_list))
        sel = faiss.IDSelectorArray(chunk_id_list)
        params = faiss.SearchParameters()
        params.sel = sel
        return dense_retrieval_subqueries_for_finetune(queries, all_queries_list, sub_queries_index, faiss_index_chunk, all_chucks, top_k=5,params=params)


if __name__ == "__main__":
    data =  {
        "question": "What spell does Harry use to save himself and Dudley from Dementors?",
        "answer": "Expecto Patronum",
        "list of reference": [
            {
                "ref_id": 4176,
                "passage": "\"Expecto Patronum!\" A silvery wisp of vapor shot from the tip of the wand and the dementor slowed, but the spell hadn't worked properly; tripping over his feet, Harry retreated farther as the dementor bore down upon him, panic fogging his brain - concentrate -\nA pair of gray, slimy, scabbed hands slid from inside the dementor's robes, reaching for him. A rushing noise filled Harry's ears. \"Expecto Patronum!\" His voice sounded dim and distant. ... Another wisp of silver smoke, feebler than the last, drifted from the wand - he couldn't do it anymore, he couldn't work the spell -\nThere was laughter inside his own head, shrill, high-pitched laughter. ... He could smell the dementor's putrid, death-cold breath, filling his own lungs, drowning him - Think ... something happy. ... But there was no happiness in him. ... The dementor's icy fingers were closing on his throat - the high-pitched laughter was growing louder and louder, and a voice spoke inside his head - \"Bow to death, Harry.",
                "book": 5,
                "chapter": 1
            },
            {
                "ref_id": 4177,
                "passage": "... It might even be painless. ... I would not know. ... I have never died.",
                "book": 5,
                "chapter": 1
            },
            {
                "ref_id": 4178,
                "passage": "...\"\nHe was never going to see Ron and Hermione again -\nAnd their faces burst clearly into his mind as he fought for breath -\n\"EXPECTO PATRONUM!\" An enormous silver stag erupted from the tip of Harry's wand; its antlers caught the dementor in the place where the heart should have been; it was thrown backward, weightless as darkness, and as the stag charged, the dementor swooped away, batlike and defeated. \"THIS WAY!\" Harry shouted at the stag. Wheeling around, he sprinted down the alleyway, holding the lit wand aloft. \"DUDLEY? DUDLEY!\" He had run barely a dozen steps when he reached them: Dudley was curled on the ground, his arms clamped over his face; a second dementor was crouching low over him, gripping his wrists in its slimy hands, prizing them slowly, almost lovingly apart, lowering its hooded head toward Dudley's face as though about to kiss him. ...\n\"GET IT!\"",
                "book": 5,
                "chapter": 1
            }
        ],
        "id": 36,
        "question_variants": "Which incantation does Harry employ to protect himself and Dudley from the Dementors' influence?",
        "sub_questions": [
            "What is the name of the spell that Harry uses to protect himself and Dudley from the Dementors?",
            "How does this spell work to counteract the effects of the Dementors' influence?",
            "Are there any other characters in the Harry Potter series who have used this spell?"
        ],
        "category": "medium_single_labeled"
    }
    CATEGORY = data["category"]
    EMBEDDING_PATH = "../embedding"
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

    EASY_INDEX = faiss.read_index(f"embedding/BAAI/bge-base-en-v1.5_finetuned/{CATEGORY}_embeddings.index")
    EASY_ALL_SUB = retrieve_all_subqueries(f"{DATA_PATH}/QA_set/{CATEGORY}.json")
    CORPUS_EMBEDDING = faiss.read_index('embedding/BAAI/bge-base-en-v1.5_finetuned/hp_all_BAAI/bge-base-en-v1.5_finetuned.index')
    KG_EMBEDDING = faiss.read_index('embedding/BAAI/bge-base-en-v1.5_finetuned/hp_kg_BAAI/bge-base-en-v1.5_finetuned.index')
    CORPUS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"
    with open(CORPUS_FILE, 'r') as f:
        CORPUS_DATA = json.load(f)
    # result = dense_retrieval_subqueries_for_finetune(data['sub_questions'], EASY_ALL_SUB, EASY_INDEX, CORPUS_EMBEDDING,CORPUS_DATA , top_k=5)
    result = KG_dense_retrieval(data['sub_questions'], EASY_ALL_SUB, EASY_INDEX, KG_EMBEDDING, CORPUS_EMBEDDING, CORPUS_DATA, relation_to_kgid_map, top_k=5)
    result_1 = dense_retrieval_subqueries_for_finetune(data['sub_questions'], EASY_ALL_SUB, EASY_INDEX, CORPUS_EMBEDDING,CORPUS_DATA , top_k=5)
    
    result_chunk_ids = [item['chunk_id'] for item in result]
    result1_chunk_ids = [item['chunk_id'] for item in result_1]
    print("kg+retrieval: ", result_chunk_ids)
    print("retrieval: ", result1_chunk_ids)
    print(result_1)

    text_enrichment = KG_on_the_fly(result_1)
    print(text_enrichment)