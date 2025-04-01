import json
import glob
import re
# this new version of the script will shuffle the questions and answers and create a new json file with the shuffled questions and answers
# Hardcoded values
input_folder = "data/QA_set"  # Adjust path as needed
output_folder = "data/QA_set"  # Adjust path as needed
qo_folder = "data/QO_set"  # Adjust path as needed
reference_file = "data/chunked_text_all_together_cleaned.json"  # Path to the reference file

# Find all JSON files in the specified folder without _data in the name
files = glob.glob(f"{input_folder}/*.json")
files = [file for file in files if "_labeled" not in file and "_unlabeled" not in file]

qo_files = glob.glob(f"{qo_folder}/*.json")

# Load reference items from the reference JSON file and create a mapping from chunk_id to passage
try:
    with open(reference_file, 'r', encoding='utf-8') as ref_f:
        ref_data = json.load(ref_f)
        if not isinstance(ref_data, list):
            print("Reference file does not contain a list of items.")
            exit()
        # Build mapping: chunk_id -> passage
        ref_mapping = {
            item["chunk_id"]: item["passage"]
            for item in ref_data
            if "chunk_id" in item and "passage" in item
        }
        book_mapping = {
            item["chunk_id"]: item["title_num"]
            for item in ref_data
            if "chunk_id" in item and "title_num" in item
        }
        chapter_mapping = {
            item["chunk_id"]: item["chapter_num"]
            for item in ref_data
            if "chunk_id" in item and "chapter_num" in item
        }
except Exception as e:
    print(f"Error reading reference file {reference_file}: {e}")
    exit()

# Load items from each JSON file
for file in files:
    try:
        qo_file = f"{qo_folder}/{file.split('/')[-1].replace('.json', '_QO.json')}"
        with open(file, 'r', encoding='utf-8') as f, open(qo_file, 'r', encoding='utf-8') as qo_f:
            raw_data = json.load(f).copy()
            bEasy = 'easy' in file
            if not bEasy:
                qo_data = json.load(qo_f).copy()
            else:
                qo_data = []
            data = []
            unlabled_data = []
            
            if not bEasy:
                assert len(raw_data) == len(qo_data), f"Length mismatch between {file} and {qo_file}"
            if isinstance(raw_data, list):
                idx_labeled = 1
                idx_unlabeled = 1
                for i, item in enumerate(raw_data):
                    references = item.get("list of reference", [])
                    if isinstance(references, list) and (len(references) == 0 or isinstance(references[0], int)):
                        # labeled data
                        actual_references = []
                        for ref_id in references:
                            passage = ref_mapping.get(ref_id)
                            book_num = book_mapping.get(ref_id)
                            chapter_num = chapter_mapping.get(ref_id)
                            if passage and book_num and chapter_num:
                                actual_references.append({"ref_id": ref_id, "passage": passage, "book": book_num, "chapter": chapter_num})
                            else:
                                raise ValueError(f"Reference id {ref_id} not found in the reference mapping.")
                        item["list of reference"] = actual_references
                        if not bEasy:
                            qo_item = qo_data[i]
                            original_question = qo_item.get("original_query")
                            variants = qo_item.get("query_variants")
                            sub_questions = qo_item.get("sub_questions")
                            assert item['question'] == original_question, f"Question mismatch at index {i} in {file}\nOriginal: {item['question']}\nQO: {original_question}"
                            item['question_variants'] = variants[0]
                            item['sub_questions'] = [re.sub(r'^\d+\.\s*', '', sub_q) for sub_q in sub_questions]
                        else:
                            item['question_variants'] =  item['question']
                            item['sub_questions'] = [item['question']]
                        item['category'] = file.split('/')[-1].split('.')[0] + "_labeled"
                        item['id'] = idx_labeled
                        idx_labeled += 1
                        data.append(item)
                    else:
                        # unlabeled data
                        if not bEasy:
                            qo_item = qo_data[i]
                            original_question = qo_item.get("original_query")
                            variants = qo_item.get("query_variants")
                            sub_questions = qo_item.get("sub_questions")
                            assert item['question'] == original_question, f"Question mismatch at index {i} in {file}"
                            item['question_variants'] = variants[0]
                            item['sub_questions'] = [re.sub(r'^\d+\.\s*', '', sub_q) for sub_q in sub_questions]
                        else:
                            item['question_variants'] =  item['question']
                            item['sub_questions'] = [item['question']]
                        item['category'] = file.split('/')[-1].split('.')[0] + "_unlabeled"
                        item['id'] = idx_unlabeled
                        idx_unlabeled += 1
                        unlabled_data.append(item)
                with open(f"{output_folder}/{file.split('/')[-1].replace('.json', '_labeled.json')}", 'w', encoding='utf-8') as out_f:
                    json.dump(data, out_f, indent=4, ensure_ascii=False)
                with open(f"{output_folder}/{file.split('/')[-1].replace('.json', '_unlabeled.json')}", 'w', encoding='utf-8') as out_unlabelf:
                    json.dump(unlabled_data, out_unlabelf, indent=4, ensure_ascii=False)
            else:
                print(f"Warning: File {file} does not contain a list. Skipping.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

