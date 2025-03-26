from utils import *
from retrieval import *
from dotenv import load_dotenv
from generation.pythia_generation import generation_answer, load_model_and_tokenizer


if __name__ == "__main__":
    query1 = "Which chess piece's movement in the underground chamber mirrors a character's sacrifice in the final book?"

    ## QA
    load_dotenv("key.env")
    api_key = os.environ.get('COHERE_API_KEY')
    llm = MyLLM(api_key=api_key)
    qo_a = QOAdvanced(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)
    sub_questions = qo_a.sub_questions(query1)

    print("Sub-Questions:")
    print(sub_questions)


    ## Retrieval
    # DATA_PATH = "./data"
    # QA_FILE = f"{DATA_PATH}/QA_set/easy_single.json"
    # ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"


    # vectorizer = TfidfVectorizer(stop_words='english')
    # doc_matrix = vectorizer.fit_transform(passages)
    passages, chunk_ids = load_passages_and_chunk_ids()
    tokenized_passages = [word_tokenize(doc.lower()) for doc in passages]
    bm25 = BM25Okapi(tokenized_passages)

    results = bm25_retrieval_subqueries(sub_questions, bm25, passages, chunk_ids, top_k=5)

    # for item in results:
    #     print(item, '\n\n')

    print(type(results))

    

    #Generate Answer
    try:
        model, tokenizer2 = load_model_and_tokenizer()
        generation_answer(sub_questions, results, model, tokenizer2)
    except Exception as e:
        print("Generation failed:", e)