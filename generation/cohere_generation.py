from dotenv import load_dotenv
import os
import cohere
from pathlib import Path

# Load dotenv path
project_root = Path.cwd().parent  
dotenv_path = project_root / "key.env"
print(f"dotenv_path: {dotenv_path}")

class CohereGenerator:
    def __init__(self, api_key=None, model="command-r-plus"):
        # Load API key from key.env if not provided
        if api_key is None:
            load_dotenv(dotenv_path=dotenv_path)
            api_key = os.getenv("COHERE_API_KEY")
            print('Cohere API key loaded from key.env')
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def generation_answer(self, origin_question, subquestions, retrieval_results, top_k=5, max_tokens=60):
        
        previous_qa = []

        if len(subquestions) > 1:
            for i, sub_query in enumerate(subquestions):
                # Get top passages for this sub-question
                top_chunks = [chunk for chunk in retrieval_results if chunk['sub_query'] == sub_query][:top_k]
                context = "\n".join(
                    [f"- Passage {j+1}: {chunk['passage']}" for j, chunk in enumerate(top_chunks)]
                )

                # Build prompt
                if previous_qa:
                    prev_qa_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in previous_qa])
                    prompt = (
                        "You are an assistant with expert knowledge of the Harry Potter series.\n"
                        "Use the previous answers and the context below to answer the next question concisely.\n\n"
                        f"Previous Q&A:\n{prev_qa_str}\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {sub_query}\n\nAnswer:"
                    )
                else:
                    prompt = (
                        "You are an assistant with expert knowledge of the Harry Potter series.\n"
                        "Answer the question concisely.\n\n"
                        f"Question: {sub_query}\n\n"
                        f"Context:\n{context}\n\nAnswer:"
                    )

                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.message.content[0].text.strip()
                previous_qa.append((sub_query, answer))

            # Build final prompt using all previous Q&A
            prev_qa_str_final = "\n".join([f"Q: {q}\nA: {a}" for q, a in previous_qa])
            prompt = (
                "You are an assistant with expert knowledge of the Harry Potter series.\n"
                "Use the previous answers and the context below to answer the question concisely.\n\n"
                f"Question: {origin_question}\n\n"
                f"Previous Q&A:\n{prev_qa_str_final}\n\nAnswer:"
            )

            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            final_answer = response.message.content[0].text.strip()

        else:
            # Only one sub-question, generate directly
            top_chunks = [chunk for chunk in retrieval_results if chunk['sub_query'] == origin_question][:top_k]
            context = "\n".join(
                    [f"- Passage {j+1}: {chunk['passage']}" for j, chunk in enumerate(top_chunks)]
                )
            
            prompt = (
                "You are a helpful assistant specializing in answering questions about the Harry Potter series.\n"
                "Provide concise and accurate answers based on the input you receive.\n\n"
                f"Question: {origin_question}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer:"
            )

            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            final_answer = response.message.content[0].text.strip()

        return previous_qa, final_answer