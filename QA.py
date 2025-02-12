from transformers import AutoModelForQuestionAnswering, TrainingArguments
from datasets import load_dataset
from utils import QA

# Load your dataset (assume a JSON structure or use a custom loading script)
dataset = load_dataset("json", data_files={
    "train": "train.json",
    "validation": "val.json"
})

model_name = "bert-base-uncased"

qa_task = QA(model_name)

training_args = TrainingArguments(
    output_dir="./model_qa",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

qa_task.train(dataset, training_args, AutoModelForQuestionAnswering)