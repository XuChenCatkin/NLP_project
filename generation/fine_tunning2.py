from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
import pandas as pd
import numpy as np
import torch
import json
import re
import os


file_path = "data/QA_set_Clean/shuffled_QA.json"
with open( file_path,'r') as f:
  data = json.load(f)

# Create a Hugging Face Dataset from the list of dictionaries
dataset = Dataset.from_list(data)

# Function to create a prompt template for each example
def format_prompt(example):
    # Ensure references are in string format
    refs = example["list of reference"]
    if isinstance(refs, list):
        refs = "\n\n".join(map(str, refs))
    # Create a prompt that includes the question, answer, and references
    example["text"] = (
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}\n"
        f"References: {refs}"
    )
    return example

# Apply the formatting function to all examples
dataset = dataset.map(format_prompt)

# Optionally remove the original columns if you only want the final text
dataset = dataset.remove_columns(["question", "answer", "list of reference", "id"])

torch.cuda.empty_cache()


os.environ["WANDB_API_KEY"] = "d0d82207708799021609c1bfb5dcfe366e027d2d"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Optional: Initialize wandb (Trainer can also handle this automatically if report_to is set)
# wandb.login() # remove the explicit login
wandb.init(project="pythia_finetuning", entity="aaron-cui990810-ucl")

# Clear any cached memory on the GPU


# Set your Pythia model identifier (example model used below)

model_name = "EleutherAI/pythia-1b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to the EOS token

if torch.cuda.is_available():
    model = model.to("cuda")
    print("Using CUDA")


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# Tokenize and remove the original text column
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up data collator for causal language modeling (mlm=False)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments with wandb logging and Hugging Face Hub integration
training_args = TrainingArguments(
    output_dir="./pythia-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=500,
    report_to=["wandb"],  # Enable wandb logging
    push_to_hub=True,     # Automatically push to the Hub after training
    hub_model_id="kea1de/pythia-finetuned",  # Change this to your HF repo name
    fp16=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start fine-tuning
trainer.train()

# After training, push the final model to Hugging Face Hub
trainer.push_to_hub()