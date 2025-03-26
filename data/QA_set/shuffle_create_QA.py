import json
import random
import glob

# Hardcoded values
input_folder = "NLP_project/data/QA_set_Clean"  # Adjust path as needed

output_file = "NLP_project/data/shuffled_QA.json"  # Adjust path as needed

output_file = "NLP_project/data/QA_set_Clean/shuffled_QA.json"  # Adjust path as needed

num_questions = 60
reference_file = "NLP_project/data/chunked_text_all_together_cleaned.json"  # Path to the reference file

# Find all JSON files in the specified folder
files = glob.glob(f"{input_folder}/*.json")

all_items = []
# count the number of items in each file, and the total number of items. return file name and number of items
for file in files:
    with open(file , 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            all_items.extend(data)
        else:
            print(f"Warning: File {file} does not contain a list. Skipping.")
    print(f"Number of items in {file}: {len(data)}")
print(f"Total number of items in the folder: {len(all_items)}")


all_items = []


# Load items from each JSON file
for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_items.extend(data)
            else:
                print(f"Warning: File {file} does not contain a list. Skipping.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not all_items:
    print("No valid items found in the provided JSON files.")
    exit()

# Shuffle the combined list
random.shuffle(all_items)

# Control the total number of questions in the output
if num_questions > len(all_items):
    print(f"Requested {num_questions} questions, but only found {len(all_items)} items. Using all available items.")
    num_questions = len(all_items)

selected_items = all_items[:num_questions]

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
except Exception as e:
    print(f"Error reading reference file {reference_file}: {e}")
    exit()

# Replace reference IDs with a list of dictionaries where each contains a numeric ref_id and its passage
for item in selected_items:
    references = item.get("list of reference", [])
    actual_references = []
    for ref_id in references:
        passage = ref_mapping.get(ref_id)
        if passage:
            actual_references.append({"ref_id": ref_id, "passage": passage})
        else:
            actual_references.append({"ref_id": ref_id, "passage": f"Reference id {ref_id} not found"})
    item["list of reference"] = actual_references

# Replace reference IDs with actual passages in each question
for item in selected_items:
    references = item.get("list of reference", [])
    actual_passages = []
    for ref_id in references:
        passage = ref_mapping.get(ref_id)
        if passage:
            actual_passages.append(passage)
        else:
            actual_passages.append(f"Reference id {ref_id} not found")
    item["list of reference"] = actual_passages
# Write the selected items to the output JSON file
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(selected_items, out_f, indent=4, ensure_ascii=False)

print(f"Output file '{output_file}' created with {len(selected_items)} questions.")

