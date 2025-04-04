import json
from openai import OpenAI
from string import Template
from collections import defaultdict



client = OpenAI(
    api_key="sk-2367b265559a4ae6b607bff8755ef431",
    base_url="https://api.deepseek.com",
)


def process_gpt(query):
    
    system_prompt = """
    You are an expert in entity extraction.
    """
    
    with open('./data/HP_KG_5_chunks/Node.json', 'r') as file:
        node_data = json.load(file)
    
    with open('./data/HP_KG_5_chunks/Special.json', 'r') as file:
        magic_data = json.load(file)

    all_entities = {item['name']: item['id'] for item in node_data+magic_data}
    entity_names = list(all_entities.keys())
    
    prompt_template = Template("""
        You are an expert in entity recognition.

        DO NOT answer the question — only pull out names that appear in the query from the provided list.

        Return a Python list of matching full names (exact or partial matches) — nothing else.
    
        Given a query, identify any matching names from the list below. Matching should be:
        - Match should be case-insensitive.
        - If the query says "Harry", and "Harry Potter" is in the entity list, include "Harry Potter".
        - If the query says "Professor [LastName]", look for any full name in the entity list that ends with [LastName], and return the full name instead.
        - You MUST include spell and potion names if they are mentioned
        - Return only names that are exactly in the entity list.
        - Return only a valid Python list of matched names, like ["Harry Potter", "Sirius Black"]

        Entities:
        $entity_names

        Query:
        "$query"
    """)
    
    user_prompt = prompt_template.substitute(
        entity_names="\n".join(entity_names),
        query=query
    ) 
    
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    extracted_names = json.loads(completion.choices[0].message.content.strip())
    print(extracted_names)
    
    matched_ids = [all_entities[name] for name in extracted_names if name in all_entities]
    return matched_ids

def find_chunk_id(target_ids):
    with open("./data/Node_Dictionary.json", "r") as f1:
        node_dict = json.load(f1)

    with open("./data/Special_Dictionary.json", "r") as f2:
        spec_dict = json.load(f2)
    
    full_dict = node_dict.copy()
    full_dict.update(spec_dict)
    
    scene_counts = defaultdict(int)
    num_entities = 0     
    for eid in target_ids:
        entity = full_dict.get(eid)
        if entity:
            appears = entity.get("list of appear", [])
            for scene_id in appears:
                scene_counts[int(scene_id)] += 1
            num_entities += 1
    shared_scenes = sorted([scene_id for scene_id, count in scene_counts.items() if count == num_entities])

    return shared_scenes
    

# query = "Which object are Harry, Ron, and Hermione searching for inside Bellatrix Lestrange's vault at Gringotts?"

# entities = process_gpt(query)
# print(entities)

# shared_chunk = find_chunk_id(entities)
# all_chunks = []
# for i in shared_chunk:
#     all_chunks+=[j for j in range(5*i, 5*i+5)]

# print(all_chunks)

# if __name__ == "__main__":
#     data = {
#         "question": "On which street do the Dursleys live at the beginning of the story?",
#         "answer": "They live at Number Four, Privet Drive.",
#         "list of reference": [
#             {
#                 "ref_id": 1,
#                 "passage": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors.",
#                 "book": 1,
#                 "chapter": 1
#             }
#         ],
#         "id": 1,
#         "question_variants": "On which street do the Dursleys live at the beginning of the story?",
#         "sub_questions": [
#             "On which street do the Dursleys live at the beginning of the story?"
#         ],
#         "category": "easy_single_labeled"
#     }

#     for i in data['sub_questions']:
#         entities = process_gpt(i)
#         print(entities)
