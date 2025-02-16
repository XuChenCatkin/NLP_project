from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Define paths
base_model = "bert-base-uncased"  # Change this to the original model you fine-tuned
model_path = "ner_model/checkpoint-2634"

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load fine-tuned model
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create the NER pipeline
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Run inference
text = "Barack Obama was the 44th President of the United States."
ner_results = nlp_ner(text)

# Print results
for entity in ner_results:
    print(entity)