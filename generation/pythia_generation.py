import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer


def embed_text(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt")  # PyTorch tensors
        with torch.no_grad():  # No gradient computation needed
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

        # Average pooling over the sequence length dimension (dim=1)
        sentence_embedding = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        return sentence_embedding  # Tensor of shape: (1, hidden_dim)



def load_model_and_tokenizer():
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1.4b-deduped",
        revision="step143000",
        cache_dir="./generation/pythia-1b-deduped/step143000",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-1.4b-deduped",
        revision="step143000",
        cache_dir="./generation/pythia-1b-deduped/step143000",
    )
    model.eval()
    return model, tokenizer

def generation_answer(subquestions, retrieval_results, model, tokenizer, top_k=5, max_new_tokens=60):
    previous_qa = []

    for i, sub_query in enumerate(subquestions):
        print(f"\nProcessing Q{i+1}: {sub_query}")

        # Get top passages for this sub-question
        top_chunks = [chunk for chunk in retrieval_results if chunk['sub_query'] == sub_query][:top_k]

        # Build context string
        context = "\n".join(
            [f"- Passage {j+1}: {chunk['passage']}" for j, chunk in enumerate(top_chunks)]
        )

        # Build prompt
        if previous_qa:
            prev_qa_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in previous_qa])
            prompt = f"""You are an assistant with expert knowledge of the Harry Potter series.
Use the previous answers and the context below to answer the next question concisely.

Previous Q&A:
{prev_qa_str}

Question: {sub_query}

Context:
{context}

Answer:"""
        else:
            prompt = f"""You are an assistant with expert knowledge of the Harry Potter series.
Answer the question below using the provided context.

Question: {sub_query}

Context:
{context}

Answer:"""

        # Tokenize input
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("Answer:")[-1].strip()

        print(f"Q{i+1}: {sub_query}")
        print(f"A{i+1}: {answer}")
        previous_qa.append((sub_query, answer))

    return previous_qa


    