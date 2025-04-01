import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.datasets import NoDuplicatesDataLoader
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datetime import datetime
from huggingface_hub import notebook_login
import wandb
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator
import logging
from utils import create_logger
import time
from retrieval import dense_retrieval_subqueries_for_finetune, retrieve_all_subqueries
import faiss
from typing import List



notebook_login()

# File Paths
DATA_PATH = "./data"
QA_PATH = f"{DATA_PATH}/QA_set"
QA_EMBEDDED_PATH = f"{DATA_PATH}/QA_set_embedded"

EASY = f"{QA_PATH}/easy_single_labeled.json"
MEDIUM_S = f"{QA_PATH}/medium_single_labeled.json"
MEDIUM_M = f"{QA_PATH}/medium_multi_labeled.json"
HARD_S = f"{QA_PATH}/hard_single_labeled.json"
HARD_M = f"{QA_PATH}/hard_multi_labeled.json"

CORPUS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"

# ALL subquery List
subqueries_easy = retrieve_all_subqueries(EASY)
subqueries_medium_s = retrieve_all_subqueries(MEDIUM_S)
subqueries_medium_m = retrieve_all_subqueries(MEDIUM_M)
subqueries_hard_s = retrieve_all_subqueries(HARD_S)
subqueries_hard_m = retrieve_all_subqueries(HARD_M)

# ALL Index Files
index_easy = faiss.read_index(f"{QA_EMBEDDED_PATH}/bge_easy_single_labeled.index")
index_medium_s = faiss.read_index(f"{QA_EMBEDDED_PATH}/bge_medium_single_labeled.index")
index_medium_m = faiss.read_index(f"{QA_EMBEDDED_PATH}/bge_medium_multi_labeled.index")
index_hard_s = faiss.read_index(f"{QA_EMBEDDED_PATH}/bge_hard_single_labeled.index")
index_hard_m = faiss.read_index(f"{QA_EMBEDDED_PATH}/bge_hard_multi_labeled.index")

# Load all bge embedding
corpus_index = faiss.read_index('hp_all_bge.index')
with open(CORPUS_FILE, 'r') as f:
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
        



model_list = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-large-en",
    "all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v3",
    "sentence-transformers/all-MiniLM-L12-v3",
    "Alibaba-NLP/gte-base-en-v1.5"
]

args = {
    "model_name": "BAAI/bge-large-en",
    "corpus_file": CORPUS_FILE,
    "easy_file": EASY,
    "medium_single_file": MEDIUM_S,
    "medium_multi_file": MEDIUM_M,
    "hard_single_file": HARD_S,
    "hard_multi_file": HARD_M,
    "batch_size": 4,
    "huggingfaceusername": "CatkinChen",
    "wandbusername": "aaron-cui990810-ucl",
    "epochs": 5,
    "margin": 0.3,
    "test_size": 0.2,
    "random_state": 42,
    "top_k": 5,
}

def parse_args():
    this_args = args.copy()
    this_args['project'] = f"{this_args['model_name'].split('/')[-1]}-finetune-project"
    this_args['experiment_name'] = f"{this_args['model_name'].split('/')[-1]}-finetune-run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    this_args["model_repo_id"] = f"{this_args['huggingfaceusername']}/{this_args['model_name'].replace('/', '_')}_retrieval_finetuned_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    simple_name_args = SimpleNamespace(**this_args)
    return simple_name_args

def load_json_data(path):
    """Load JSON data from a file."""
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
    """Randomly sample n elements from a list."""
    if len(data) < n:
        print(f"Warning: Not enough data to sample {n} elements. Returning all available data.")
        return data
    return np.random.choice(data, n, replace=False).tolist()

def hard_negative_mining(item):
    """Perform hard negative mining on a single item."""
    try:
        references = item["list of reference"]
        reference_size = len(references)
        if reference_size == 0:
            reference_ids = []
        else:
            reference_ids = [ref["ref_id"] for ref in references]
        reference_ids = set(reference_ids)
        sub_questions = item["sub_questions"]
        index_file, all_subquestion_list = load_index_and_all_subqueries(item["category"])
        negative_retrieval_set = set()
        top_k_retrieval = dense_retrieval_subqueries_for_finetune(sub_questions, all_subquestion_list, index_file, corpus_index, corpus, top_k=args.top_k)
        negative_retrieval_set.update([negative_retrieval['chunk_id'] for negative_retrieval in top_k_retrieval if negative_retrieval['chunk_id'] not in reference_ids])
        negative_retrieval_list = list(negative_retrieval_set)
        negative_retrievals = [corpus[int(neg_id) - 1] for neg_id in negative_retrieval_list]
        assert len(negative_retrievals) >= 1, f"Not enough negative samples found for item: {item}"
        return negative_retrievals
    except KeyError:
        raise (f"KeyError: 'list of reference' not found in item: {item}")
    except Exception as e:
        raise (f"Unexpected error: {e}")
    

class TimedCallback:
    def __init__(self):
        self.last_time = None

    def __call__(self, loss, epoch, step):
        now = time.time()
        if self.last_time is None:
            step_time = 0.0
        else:
            step_time = now - self.last_time
        self.last_time = now
        
        wandb.log({
            "train_loss": loss,
            "epoch": epoch,
            "train_step": step,
            "step_time_seconds": step_time
        })
        
def process_data(data):
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
        # negative_samples = np.random.choice(np.array(negative_list), min(1, ref_len), replace=True)

        for i in range(ref_len):
            if len(refs) == 0:
                positive_enhanced = ''
            else:
                pos_book_id = refs[i]['book']
                pos_chapter_id = refs[i]['chapter']
                passage_text = refs[i]['passage']
                positive_enhanced = (
                    # f"Book: {pos_book_id}, Chapter: {pos_chapter_id}\n"
                    f"Passage: {passage_text}"
                )
            for j in range(len(negative_list)):
                neg_book_id = negative_list[j]['title_num']
                neg_chapter_id = negative_list[j]['chapter_num']
                neg_passage_text = negative_list[j]['passage']
                negative_enhanced = (
                    # f"Book: {neg_book_id}, Chapter: {neg_chapter_id}\n"
                    f"Passage: {neg_passage_text}"
                )
                examples.append(InputExample(texts=[question, positive_enhanced, negative_enhanced]))
        # for i in range(len(negative_samples)):
        #     if len(refs) == 0:
        #         positive_enhanced = ''
        #     else:
        #         pos_book_id = refs[i]['book']
        #         pos_chapter_id = refs[i]['chapter']
        #         passage_text = refs[i]['passage']
        #         positive_enhanced = (
        #             f"Book: {pos_book_id}, Chapter: {pos_chapter_id}\n"
        #             f"Passage: {passage_text}"
        #         )
        #     neg_book_id = negative_samples[i]['title_num']
        #     neg_chapter_id = negative_samples[i]['chapter_num']
        #     neg_passage_text = negative_samples[i]['passage']
        #     negative_enhanced = (
        #         f"Book: {neg_book_id}, Chapter: {neg_chapter_id}\n"
        #         f"Passage: {neg_passage_text}"
        #     )
        #     examples.append(InputExample(texts=[question, positive_enhanced, negative_enhanced]))
    return examples, query_map, relevant_map


def process_data_MNRL(data):
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
        # negative_samples = np.random.choice(np.array(negative_list), min(1, ref_len), replace=True)

        for i in range(ref_len):
            if len(refs) == 0:
                positive_enhanced = ''
            else:
                pos_book_id = refs[i]['book']
                pos_chapter_id = refs[i]['chapter']
                passage_text = refs[i]['passage']
                positive_enhanced = (
                    f"Book: {pos_book_id}, Chapter: {pos_chapter_id}\n"
                    f"Passage: {passage_text}"
                )
            negative_enhanced = []
            for j in range(len(negative_list)):
                neg_book_id = negative_list[j]['title_num']
                neg_chapter_id = negative_list[j]['chapter_num']
                neg_passage_text = negative_list[j]['passage']
                negative_enhanced.append(
                    f"Book: {neg_book_id}, Chapter: {neg_chapter_id}\n"
                    f"Passage: {neg_passage_text}"
                )
            examples.append(InputExample(texts=[question, positive_enhanced]+ negative_enhanced))
    return examples, query_map, relevant_map



    
def train(args, logger: logging.Logger):
    # Use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load models on GPU
    logger.info(f"Loading model {args.model_name} on {device}")
    model = SentenceTransformer(args.model_name, device=device)
    
    logger.info(f"Loading data")
    corpus = load_json_data(args.corpus_file)
    corpus_map = dict([(str(item["chunk_id"]), item['passage']) for item in corpus])
    easy = load_json_data(args.easy_file)
    medium_single = load_json_data(args.medium_single_file)
    medium_multi = load_json_data(args.medium_multi_file)
    hard_single = load_json_data(args.hard_single_file)
    hard_multi = load_json_data(args.hard_multi_file)

    train_data_easy, test_data_easy = train_test_split(easy, test_size=args.test_size, random_state=args.random_state)
    train_data_medium_single, test_data_medium_single = train_test_split(medium_single, test_size=args.test_size, random_state=args.random_state)
    train_data_medium_multi, test_data_medium_multi = train_test_split(medium_multi, test_size=args.test_size, random_state=args.random_state)
    train_data_hard_single, test_data_hard_single = train_test_split(hard_single, test_size=args.test_size, random_state=args.random_state)
    train_data_hard_multi, test_data_hard_multi = train_test_split(hard_multi, test_size=args.test_size, random_state=args.random_state)
    train_data = train_data_easy + train_data_medium_single + train_data_medium_multi + train_data_hard_single + train_data_hard_multi
    test_data = test_data_easy + test_data_medium_single + test_data_medium_multi + test_data_hard_single + test_data_hard_multi
    
    # save down train and test data
    with open(f"data/finetune_train_data_test_size_{args.test_size}_random_state_{args.random_state}.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open(f"data/finetune_test_data_test_size_{args.test_size}_random_state_{args.random_state}.json", "w") as f:
        json.dump(test_data, f, indent=4)
    logger.info(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")

    train_examples, train_query_map, train_relevant_map = process_data_MNRL(train_data)
    train_examples_dict = [ {"question": example.texts[0], "positive": example.texts[1], "negative": example.texts[2]} for example in train_examples ]

    # train_examples, train_query_map, train_relevant_map = process_data(train_data)
    # train_examples_dict = [ {"question": example.texts[0], "positive": example.texts[1], "negative": example.texts[2]} for example in train_examples ]



    with open (f"data/finetune_train_triplets_test_size_{args.test_size}_random_state_{args.random_state}_processed.json", "w") as f:
        json.dump(train_examples_dict, f, indent=4)
    logger.info(f"Processed {len(train_examples)} training examples")
    test_examples, test_query_map, test_relevant_map = process_data_MNRL(test_data)
    # test_examples, test_query_map, test_relevant_map = process_data(test_data)

    
   
    train_dataloader = NoDuplicatesDataLoader(train_examples,batch_size=args.batch_size)
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    evaluator = InformationRetrievalEvaluator(
        test_query_map,
        corpus_map,
        test_relevant_map,
        len(corpus_map)
    )

    # train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=args.margin)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    logger.info(f"Training with {len(train_examples)} training examples and {len(test_examples)} test examples")

    wandb.config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_name": args.model_name,
        "train_data_size": len(train_examples),
        "test_data_size": len(test_examples),
        "margin": args.margin
    })

    logger.info(f"Training model")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=2,
        evaluator=evaluator,
        evaluation_steps=5,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True,
        callback=TimedCallback()
    )
    
    logger.info(f"Evaluating model")

    eval_score = evaluator(model, output_path=None)
    wandb.log({
        "eval_score": eval_score,
        "epoch": args.epochs
    })

    wandb.finish()

    logger.info(f"Saving model to Hugging Face Hub")
    
    model.push_to_hub(
        repo_id=args.model_repo_id
    )
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    wandb.init(project=args.project, name=args.experiment_name, entity=args.wandbusername)
    logger = create_logger(args.experiment_name, console_output=True)
    logger.info(f"Using arguments: {args}")
    train(args, logger)