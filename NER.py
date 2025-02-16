from transformers import AutoModelForTokenClassification, TrainingArguments
from datasets import load_dataset
from utils import NER

# Load your dataset
dataset = load_dataset(
#     'json', data_files={
#     'train': 'train.json',
#     'validation': 'validation.json'
# }
"conll2003"
)

# Create a label-to-ID mapping
# In this example, we have 9 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
# O is the label for tokens that are not part of any named entity
# B-XXX is the label for the first token of a named entity of type XXX
# I-XXX is the label for the second and subsequent tokens of a named entity of type XXX
# For example, "John Doe" would be annotated as ["B-PER", "I-PER"]
# PER, ORG, LOC, and MISC mean person, organization, location, and miscellaneous, respectively
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

model_name = "bert-base-uncased"

ner_task = NER(model_name, label_list)

# Training parameters
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Train the NER model
ner_task.train(dataset, training_args, AutoModelForTokenClassification)
results = ner_task.eval()