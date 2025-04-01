import json
import os
from utils import QOAdvanced
import cohere
from dotenv import load_dotenv

# Load environment variables (your COHERE API key)
load_dotenv("/Users/yandu/NLP_project/key.env")
api_key = os.environ.get('COHERE_API_KEY')

# Dummy LLM class as in your snippet
import time

class MyLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Please set the COHERE_API_KEY environment variable")

    def generate(self, prompt: str) -> str:
        # Add a 1-second delay to help avoid rate-limit issues
        time.sleep(1)
        
        co = cohere.ClientV2(api_key=self.api_key)
        res = co.chat(
            model="command-r7b-12-2024",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return res.message.content[0].text


def process_and_save_queries(input_file: str, output_file: str):
    """
    Reads queries from the input JSON file, uses QOAdvanced methods to generate
    query variants and sub-questions, and writes the results to the output JSON file.
    """
    # Initialize the LLM and QOAdvanced instance
    llm = MyLLM(api_key)
    qo_a = QOAdvanced(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)
    
    # Load the queries from the input JSON file
    with open(input_file, 'r') as f:
        queries_data = json.load(f)
    
    results = []
    
    # Process each query entry in the file
    for entry in queries_data:
        query = entry.get("question", "")
        id = entry.get("id")
        if not query:
            continue  # Skip if no query is provided
        
        # Generate query variants using multi_query
        query_variants = qo_a.multi_query(query)
        
        # Generate sub-questions for the given query
        sub_questions = qo_a.sub_questions(query)
        
        # Append the results (you can include additional metadata if needed)
        results.append({
            "original_query": query,
            "query_variants": query_variants,
            "sub_questions": sub_questions,
            "id": id
        })
    
    # Write the generated queries to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Example usage:
if __name__ == "__main__":
    import os

    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # If the file is in the same directory:
    # input_file = os.path.join(current_dir, "easy_single_reindexed.json")
    output_file = os.path.join(current_dir, "data/QO_set/medium_multi_QO_new.json")

    # If the file is in the parent directory, uncomment the following:
    input_file = os.path.join(current_dir, "data/QA_set/medium_multi.json")

    # Process the queries using the function
    process_and_save_queries(input_file, output_file)
    print(f"Processed queries have been saved to {output_file}")

