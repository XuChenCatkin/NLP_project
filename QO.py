# Import the QO class from utils.py
from utils import QO
from utils import QOAdvanced
import cohere
from dotenv import load_dotenv
import os

load_dotenv("key.env")  # Loads variables from .env into the environment
# print(os.environ) 
api_key = os.environ.get('COHERE_API_KEY')
# print(api_key)  # Verify that the key is loaded

# Assume we have an LLM instance that supports a .generate() method.
# This could be any language model interface you have implemented.
# For demonstration purposes, we create a dummy LLM:
class MyLLM:
    def __init__(self):
        # Retrieve the API key from the environment variable.
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Please set the COHERE_API_KEY environment variable")

    def generate(self, prompt: str) -> str:
        co = cohere.ClientV2(api_key=self.api_key)
        res = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return res.message.content[0].text



# Create an instance of your dummy LLM and the QO class
llm = MyLLM()
# qo = QO(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)
qo_a = QOAdvanced(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)

# Example query needs optimization
# query = "When was Hogwarts founded and who founded it?"
# query = 'Which of the following is NOT one of Albus Dumbledore\'s titles?'
# query = 'Who teaches History of Magic at Hogwarts?'
# query = 'Did Harry Potter defeat lord Voldemort?'      #######################################################
query = "What subtle hint in a conversation about a magical creature suggests a future betrayal in the first book?"

# --- Query Transformation ---
# 1. Use HyDE to generate an embedding from a hypothetical document
try:
    embedding = qo_a.hyde(query)
    print("HyDE Embedding:", embedding)
    print('Embedding Successful!')
except ValueError as e:
    print(e)

# 2. Use Step-Back Prompting to generate a more abstract version of the query
# abstract_query = qo.step_back(query)
# print("Abstract Query:", abstract_query)

# --- Query Routing ---
# Define available sources with metadata keywords
sources = {
    "harry_potter": {"keywords": ["hogwarts", "harry potter", "wizard"]},
    "wikipedia": {"keywords": []}  # default/general source
}
selected_source = qo_a.semantic_routing(query, sources)
print("Selected Source:", selected_source)

# --- Metadata Filtering ---
# Define available metadata for filtering (for example, mapping book IDs to titles)
available_meta = {
    "book": {
        1: "Harry Potter and the Philosopher's Stone",
        2: "Harry Potter and the Chamber of Secrets",
        3: "Harry Potter and the Prisoner of Azkaban",
        4: "Harry Potter and the Goblet of Fire",
        5: "Harry Potter and the Order of the Phoenix",
        6: "Harry Potter and the Half-Blood Prince",
        7: "Harry Potter and the Deathly Hallows"
    }
}
filters = qo_a.metadata_filter(query, available_meta)
print("Metadata Filters:", filters)

# --- Query Expansion ---
# 1. Generate multiple query variants
query_variants = qo_a.multi_query(query)
# query_variants = qo.multi_query(abstract_query)
print("Query Variants:")
for variant in query_variants:
     print(" -", variant)

# 2. Decompose the query into sub-questions if it is compound
sub_questions = qo_a.sub_questions(query)
# sub_questions = qo.sub_questions(abstract_query)

# Split by lines and search for the first variant that starts with "1."
# lines = query_variants.splitlines()
# first_variant_line = next(line for line in lines if line.strip().startswith("1."))
# Remove the numbering and quotes if needed:
# first_variant = first_variant_line.split("1.", 1)[1].strip().strip('"')
# print(first_variant)

# variants_str = "\n".join(query_variants)
# lines = variants_str.splitlines()
# print(lines[3][2:])

# sub_questions = qo.sub_questions(lines[3][2:])

print("Sub-Questions:")
for subq in sub_questions:
    print(" -", subq)

# The optimized queries or embeddings can now be used by the QA system (QA.py)
# or to run NER tasks (NER.py), further integrating with the overall system.
