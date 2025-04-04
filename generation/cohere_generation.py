from dotenv import load_dotenv
import os
import cohere
from pathlib import Path
project_root = Path.cwd().parent  
dotenv_path = project_root / "key.env"
print(f"dotenv_path: {dotenv_path}")
class CohereGenerator:
    def __init__(self, api_key=None, model="command-r"):
        # Load API key from key.env if not provided
        if api_key is None:
            #load_dotenv("key.env")
            load_dotenv(dotenv_path=dotenv_path)
            api_key = os.getenv("COHERE_API_KEY")
            print('Cohere API key loaded from key.env')
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def generation_answer(self, origin_question, subquestions, retrieval_results, top_k=5, max_tokens=60):
        previous_qa = []

        for i, sub_query in enumerate(subquestions):
            #print(f"\nProcessing Q{i+1}: {sub_query}")

            # Get top passages for this sub-question
            top_chunks = [chunk for chunk in retrieval_results if chunk['sub_query'] == sub_query][:top_k]

            # Build context string from the retrieved passages
            context = "\n".join(
                [f"- Passage {j+1}: {chunk['passage']}" for j, chunk in enumerate(top_chunks)]
            )

            # Build the prompt with previous Q&A if available
            if previous_qa:
                prev_qa_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in previous_qa])
                prompt = f"""You are an assistant with expert knowledge of the Harry Potter series.
Use the previous answers and the context below to answer the next question concisely.

Previous Q&A:
{prev_qa_str}

Context:
{context}

Question: {sub_query}

Answer:"""
            else:
                prompt = f"""You are an assistant with expert knowledge of the Harry Potter series.
Answer the question concisely.

Question: {sub_query}

Context:
{context}

Answer:"""

            # print("Prompt:\n", prompt)

            # Generate answer using Cohere
            response = self.client.chat(
                model=self.model,
                # prompt=prompt,
                # max_tokens=max_tokens,
                # temperature=0.3,
                # stop_sequences=["\nQ:"]
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
            )

            # print(response)
            answer = response.message.content[0].text.strip()
            # print(f"Q{i+1}: {sub_query}")
            # print(f"A{i+1}: {answer}")
            previous_qa.append((sub_query, answer))

        # print('\n\nProcess original Question')
        prev_qa_str_final = "\n".join([f"Q: {q}\nA: {a}" for q, a in previous_qa])

        prompt = f'''You are an assistant with expert knowledge of the Harry Potter series.
Use the previous answers and the context below to answer the next question concisely.

Question: {origin_question}

Previous Q&A:
{prev_qa_str_final}

Answer: 
'''
        #print(prompt)
        response = self.client.chat(
                model=self.model,
                # prompt=prompt,
                # max_tokens=max_tokens,
                # temperature=0.3,
                # stop_sequences=["\nQ:"]
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
            )
        final_answer = response.message.content[0].text.strip()
        # print(f"Question: {origin_question}")
        # print(f"Answer: {final_answer}")

        return previous_qa, final_answer