import pysbd
import json
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def convert_float32_to_float(obj):
    """
    Recursively convert any float32 or 0-dim torch Tensors to Python floats.
    """
    if isinstance(obj, dict):
        # Convert each value in a dict
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Convert each element in a list
        return [convert_float32_to_float(x) for x in obj]
    elif hasattr(obj, "item") and callable(obj.item):
        # Covers PyTorch Tensors of size 1 (e.g., torch.tensor(0.99))
        return float(obj.item())
    elif isinstance(obj, np.float32):
        return float(obj)  # convert np.float32 -> Python float
    elif isinstance(obj, float):
        # Covers standard Python floats (including float32 from some pipelines)
        return float(obj)  # ensure it's a Python float (float64)
    else:
        return obj
    
def combine_subwords(entities):
    combined_entities = []
    current_entity = None

    for entity in entities:
        word = entity['word']
        if word.startswith("##"):
            if current_entity:
                current_entity['word'] += word[2:]
                current_entity['end'] = entity['end']
        else:
            if current_entity:
                combined_entities.append(current_entity)
            current_entity = entity.copy()

    if current_entity:
        combined_entities.append(current_entity)

    return combined_entities

def process_paragraph(paragraph_lines, paragraph_index):
        """
        Takes all the lines for one paragraph, splits them into sentences,
        runs NER on each sentence, and writes a single NDJSON record with
        the combined entities for that paragraph.
        """
        # Join all lines into a single string for this paragraph
        
        nlp_ner = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        
        paragraph_text = " ".join(paragraph_lines).strip()
        if not paragraph_text:
            return  # no text to process, skip

        # Split paragraph into sentences
        segmenter = pysbd.Segmenter(language="en", clean=True)
        sentences = segmenter.segment(paragraph_text)

        # Gather entities from all sentences in this paragraph
        paragraph_entities = []
        for sent in sentences:
            # print(sent)
            sent_entities = combine_subwords(nlp_ner(sent))
            # print(sent_entities)
            sent_entities = convert_float32_to_float(sent_entities)
            paragraph_entities.extend(sent_entities)

        # Create a record with paragraph index + combined entities
        record = {
            "paragraph_index": paragraph_index,
            "entities": paragraph_entities
        }

        # Write as NDJSON (one JSON object per line)
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# Define paths
base_model = "bert-base-uncased"  # Change this to the original model you fine-tuned
model_path = "ner_model/checkpoint-2634"

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load fine-tuned model
model = AutoModelForTokenClassification.from_pretrained(model_path)

# # Create the NER pipeline
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# text = "Harry had never even imagined such a strange and splendid place. It was lit by thousands and thousands of candles that were floating in midair over four long tables, where the rest of the students were sitting. These tables were laid with glittering golden plates and goblets. At the top of the hall was another long table where the teachers were sitting. Professor McGonagall led the first years up here, so that they came to a halt in a line facing the other students, with the teachers behind them. The hundreds of faces staring at them looked like pale lanterns in the flickering candlelight. Dotted here and there among the students, the ghosts shone misty silver. Mainly to avoid all the staring eyes, Harry looked upward and saw a velvety black ceiling dotted with stars. Harry quickly looked down again as Professor McGonagall silently placed a four-legged stool in front of the first years. On top of the stool she put a pointed wizard's hat. This hat was patched and frayed and extremely dirty. Aunt Petunia wouldn't have let it in the house."

# text1 = "Harry had never even imagined such a strange and splendid place."

# text = "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense."

# ner_results = combine_subwords(nlp_ner(text))

# print(text)
# for entity in ner_results:
#     print(entity)

# Run inference
input_file = "Harry_Potter_1.txt"      
output_file = "ner_results.ndjson"

paragraph_lines = []  # Temporary storage for lines belonging to the current paragraph
paragraph_index = 0  
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    
    for line in fin:
        line_stripped = line.strip()
        
        if line_stripped == "":
            # Empty line signals the end of this paragraph
            if paragraph_lines:  # If we have accumulated lines
                process_paragraph(paragraph_lines, paragraph_index)
                paragraph_lines = []
                paragraph_index += 1
        else:
            # Accumulate lines for the current paragraph
            paragraph_lines.append(line_stripped)
    
    # If the file doesn't end with an empty line, process the last paragraph
    if paragraph_lines:
        process_paragraph(paragraph_lines, paragraph_index)

print(f"NER results per paragraph saved to: {output_file}")