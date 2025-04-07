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
    
    with open('../data/HP_KG_5_chunks/Node.json', 'r') as file:
        node_data = json.load(file)
    
    with open('../data/HP_KG_5_chunks/Special.json', 'r') as file:
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
    # print(extracted_names)
    
    matched_ids = [all_entities[name] for name in extracted_names if name in all_entities]
    return matched_ids

def find_chunk_id(target_ids):
    with open("../data/Node_Dictionary.json", "r") as f1:
        node_dict = json.load(f1)

    with open("../data/Special_Dictionary.json", "r") as f2:
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


def augment_subqueries(queries):
    system_prompt = """
    Forget everything you know about the world. You are an expert in extracting triplets from text.
    Extract informative triplets directly from the text following the examples. 
    Do not add any extra words, line breaks, or spaces and any other information.
    """
    prompt_template = Template("""
        For the text given below, you should follow the instructions to finish the task:
        1. Identify the useful entities, the entities domains are:
            - Person(the characters that appear in the story)
            - Place(the places that appear in the story)
            - Organization(the organizations that appear in the story)
            - Event(the events that happen in the story)
            - Object(the objects that appear in the story)
            - Spell(the spells that appear in the story)
            - Potion(the potions that appear in the story)
            - Creature(the creatures that appear in the story)
        2. read the text carefully and extract the triplets that identify the interactions between the entities.
            - the triplet can be in the form of (subject, predicate, object), (subject, verb, object), (subject, action, object), (subject, event, object), (subject, place, object), (subject, organization, object), (subject, person, object), (subject, creature, object), (subject, spell, object), (subject, potion, object).
        3. for the triplets you extract, group them by the following rules:
            - if the triplets same entity, group them together
        4. after you group them, you should have several groups of triplets.
        5. understand the meaning of the triplets in each group, and rephrase them into a readable sentence using only the information that given in this group of triplets, do not using any of your background knowledge about the Harry Potter.
        6. combine all the sentences you get from the groups into a paragraph, make sure the paragraph don't have any special signs like ":", "-", "=", "(", ")", make sure the paragraph is not too long.
        text:
        $text
    """)
    user_prompt = prompt_template.substitute(
        text=queries
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    complete_graph_texts = completion.choices[0].message.content.strip()
    return complete_graph_texts


def KG_on_the_fly(queries):
    origin_text = ''
    for query in queries:
        origin_text += query["passage"]
    text_completion = augment_subqueries(origin_text)
    # print("text_completion", text_completion)
    
    
    return text_completion
    

# quries = [{'chunk_id': 1195, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 12, 'chapter_name': 'The Polyjuice Potion', 'passage': '"Good thinking," said Ron, unlocking the door. "We\'ll take separate stalls." Careful not to spill a drop of his Polyjuice Potion, Harry slipped into the middle stall. "Ready?" he called. "Ready," came Ron\'s and Hermione\'s voices. "One - two - three -"\nPinching his nose, Harry drank the potion down in two large gulps.'}, {'chunk_id': 7714, 'title_num': 7, 'title': 'Harry Potter and the Deathly Hallows', 'chapter_num': 5, 'chapter_name': 'Fallen Warrior', 'passage': 'roared Hagrid. "I\'m sorry, Harry, but I had to check," said Lupin tersely. "We\'ve been betrayed. Voldemort knew that you were being moved tonight and the only people who could have told him were directly involved in the plan. You might have been an impostor." "So why aren\' you checkin\' me?" panted Hagrid, still struggling to fit through the door. "You\'re half-giant," said Lupin, looking up at Hagrid. "The Polyjuice Potion is designed for human use only."'}, {'chunk_id': 3992, 'title_num': 4, 'title': 'Harry Potter and the Goblet of Fire', 'chapter_num': 35, 'chapter_name': 'Veritaserum', 'passage': 'Then he picked up the hip flask that stood upon the desk, unscrewed it, and turned it over. A thick glutinous liquid splattered onto the office floor. "Polyjuice Potion, Harry," said Dumbledore. "You see the simplicity of it, and the brilliance. For Moody never does drink except from his hip flask, he\'s well known for it. The imposter needed, of course, to keep the real Moody close by, so that he could continue making the potion. You see his hair ..." Dumbledore looked down on the Moody in the trunk. "The imposter has been cutting it off all year, see where it is uneven?'}, {'chunk_id': 1073, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 9, 'chapter_name': 'The Writing on the Wall', 'passage': '"What we\'d need to do is to get inside the Slytherin common room and ask Malfoy a few questions without him realizing it\'s us." "But that\'s impossible," Harry said as Ron laughed. "No, it\'s not," said Hermione. "All we\'d need would be some Polyjuice Potion." "What\'s that?" said Ron and Harry together. "Snape mentioned it in class a few weeks ago -"\n"D\'you think we\'ve got nothing better to do in Potions than listen to Snape?" muttered Ron.'}, {'chunk_id': 4678, 'title_num': 5, 'title': 'Harry Potter and the Order of the Phoenix', 'chapter_num': 12, 'chapter_name': 'Professor Umbridge', 'passage': '"A light silver vapor should now be rising from your potion," called Snape, with ten minutes left to go. Harry, who was sweating profusely, looked desperately around the dungeon. His own cauldron was issuing copious amounts of dark gray steam; Ron\'s was spitting green sparks. Seamus was feverishly prodding the flames at the base of his cauldron with the tip of his wand, as they had gone out. The surface of Hermione\'s potion, however, was a shimmering mist of silver vapor, and as Snape swept by he looked down his hooked nose at it without comment, which meant that he could find nothing to criticize. At Harry\'s cauldron, however, Snape stopped, looking down at Harry with a horrible smirk on his face. "Potter, what is this supposed to be?" The Slytherins at the front of the class all looked up eagerly; they loved hearing Snape taunt Harry. "The Draught of Peace," said Harry tensely. "Tell me, Potter," said Snape softly, "can you read?" Draco Malfoy laughed. "Yes, I can," said Harry, his fingers clenched tightly around his wand. "Read the third line of the instructions for me, Potter." Harry squinted at the blackboard; it was not easy to make out the instructions through the haze of multicolored steam now filling the dungeon.\n" \'Add powdered moonstone, stir three times counterclockwise, allow to simmer for seven minutes, then add two drops of syrup of hellebore.\' "\nHis heart sank. He had not added syrup of hellebore, but had proceeded straight to the fourth line of the instructions after allowing his potion to simmer for seven minutes. "Did you do everything on the third line, Potter?" "No," said Harry very quietly. "I beg your pardon?" "No," said Harry, more loudly.'}, {'chunk_id': 1195, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 12, 'chapter_name': 'The Polyjuice Potion', 'passage': '"Good thinking," said Ron, unlocking the door. "We\'ll take separate stalls." Careful not to spill a drop of his Polyjuice Potion, Harry slipped into the middle stall. "Ready?" he called. "Ready," came Ron\'s and Hermione\'s voices. "One - two - three -"\nPinching his nose, Harry drank the potion down in two large gulps.'}, {'chunk_id': 8142, 'title_num': 7, 'title': 'Harry Potter and the Deathly Hallows', 'chapter_num': 14, 'chapter_name': 'The Thief', 'passage': 'However, in the few moments it took for Ron to give a low groan and Harry to start crawling toward him, he realized that this was not the Forbidden Forest: The trees looked younger, they were more widely spaced, the ground clearer. He met Hermione, also on her hands and knees, at Ron\'s head. The moment his eyes fell upon Ron, all other concerns fled Harry\'s mind, for blood drenched the whole of Ron\'s left side and his face stood out, grayish-white, against the leaf-strewn earth. The Polyjuice Potion was wearing off now: Ron was halfway between Cattermole and himself in appearance, his hair turning redder and redder as his face drained of the little color it had left. "What\'s happened to him?" "Splinched," said Hermione, her fingers already busy at Ron\'s sleeve, where the blood was wettest and darkest. Harry watched, horrified, as she tore open Ron\'s shirt. He had always thought of Splinching as something comical, but this ... His insides crawled unpleasantly as Hermione laid bare Ron\'s upper arm, where a great chunk of flesh was missing, scooped cleanly away as though by a knife. "Harry, quickly, in my bag, there\'s a small bottle labeled \'Essence of Dittany\' -"\n"Bag - right -"\nHarry sped to the place where Hermione had landed, seized the tiny beaded bag, and thrust his hand inside it. At once, object after object began presenting itself to his touch: He felt the leather spines of books, woolly sleeves of jumpers, heels of shoes -\n"Quickly!" He grabbed his wand from the ground and pointed it into the depths of the magical bag. "Accio Dittany!" A small brown bottle zoomed out of the bag; he caught it and hastened back to Hermione and Ron, whose eyes were now half-closed, strips of white eyeball all that were visible between his lids. "He\'s fainted," said Hermione, who was also rather pale; she no longer looked like Mafalda, though her hair was still gray in places. "Unstopper it for me, Harry, my hands are shaking." Harry wrenched the stopper off the little bottle, Hermione took it and poured three drops of the potion onto the bleeding wound. Greenish smoke billowed upward and when it had cleared, Harry saw that the bleeding had stopped. The wound now looked several days old; new skin stretched over what had just been open flesh. "Wow," said Harry. "It\'s all I feel safe doing," said Hermione shakily. "There are spells that would put him completely right, but I daren\'t try in case I do them wrong and cause more damage. ... He\'s lost so much blood already. ..."\n"How did he get hurt? I mean" - Harry shook his head, trying to clear it, to make sense of whatever had just taken place - "why are we here?'}, {'chunk_id': 1192, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 12, 'chapter_name': 'The Polyjuice Potion', 'passage': 'They could hardly see for the thick black smoke issuing from the stall in which Hermione was stirring the cauldron. Pulling their robes up over their faces, Harry and Ron knocked softly on the door. "Hermione?" They heard the scrape of the lock and Hermione emerged, shiny-faced and looking anxious. Behind her they heard the gloop gloop of the bubbling, glutinous potion. Three glass tumblers stood ready on the toilet seat. "Did you get them?" Hermione asked breathlessly. Harry showed her Goyle\'s hair.'}, {'chunk_id': 6606, 'title_num': 6, 'title': 'Harry Potter and the Half-Blood Prince', 'chapter_num': 9, 'chapter_name': 'The Half-Blood Prince', 'passage': 'Hermione\'s hand was fastest once more. "It\'s Polyjuice Potion, sir," she said. Harry too had recognized the slow-bubbling, mudlike substance in the second cauldron, but did not resent Hermione getting the credit for answering the question; she, after all, was the one who had succeeded in making it, back in their second year. "Excellent, excellent! Now, this one here ... yes, my dear?" said Slughorn, now looking slightly bemused, as Hermione\'s hand punched the air again. "It\'s Amortentia!" "It is indeed.'}, {'chunk_id': 1073, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 9, 'chapter_name': 'The Writing on the Wall', 'passage': '"What we\'d need to do is to get inside the Slytherin common room and ask Malfoy a few questions without him realizing it\'s us." "But that\'s impossible," Harry said as Ron laughed. "No, it\'s not," said Hermione. "All we\'d need would be some Polyjuice Potion." "What\'s that?" said Ron and Harry together. "Snape mentioned it in class a few weeks ago -"\n"D\'you think we\'ve got nothing better to do in Potions than listen to Snape?" muttered Ron.'}, {'chunk_id': 1217, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 12, 'chapter_name': 'The Polyjuice Potion', 'passage': 'Her face was covered in black fur. Her eyes had turned yellow and there were long, pointed ears poking through her hair. "It was a c-cat hair!" she howled. "M-Millicent Bulstrode m-must have a cat! And the p-potion isn\'t supposed to be used for animal transformations!" "Uh-oh," said Ron. "You\'ll be teased something dreadful," said Myrtle happily. "It\'s okay, Hermione," said Harry quickly. "We\'ll take you up to the hospital wing. Madam Pomfrey never asks too many questions. ..."\nIt took a long time to persuade Hermione to leave the bathroom. Moaning Myrtle sped them on their way with a hearty guffaw. "Wait till everyone finds out you\'ve got a tail!"'}, {'chunk_id': 4998, 'title_num': 5, 'title': 'Harry Potter and the Order of the Phoenix', 'chapter_num': 18, 'chapter_name': "Dumbledore's Army", 'passage': '"Fine, let\'s swap," said Hermione, seizing Ron\'s raven and replacing it with her own fat bullfrog. "Silencio!" The raven continued to open and close its sharp beak, but no sound came out. "Very good, Miss Granger!" said Professor Flitwick\'s squeaky little voice and Harry, Ron, and Hermione all jumped. "Now, let me see you try, Mr. Weasley!" "Wha - ? Oh - oh, right," said Ron, very flustered. "Er - Silencio!" He jabbed at the bullfrog so hard that he poked it in the eye; the frog gave a deafening croak and leapt off the desk.'}, {'chunk_id': 1195, 'title_num': 2, 'title': 'Harry Potter and the Chamber of Secrets', 'chapter_num': 12, 'chapter_name': 'The Polyjuice Potion', 'passage': '"Good thinking," said Ron, unlocking the door. "We\'ll take separate stalls." Careful not to spill a drop of his Polyjuice Potion, Harry slipped into the middle stall. "Ready?" he called. "Ready," came Ron\'s and Hermione\'s voices. "One - two - three -"\nPinching his nose, Harry drank the potion down in two large gulps.'}, {'chunk_id': 8142, 'title_num': 7, 'title': 'Harry Potter and the Deathly Hallows', 'chapter_num': 14, 'chapter_name': 'The Thief', 'passage': 'However, in the few moments it took for Ron to give a low groan and Harry to start crawling toward him, he realized that this was not the Forbidden Forest: The trees looked younger, they were more widely spaced, the ground clearer. He met Hermione, also on her hands and knees, at Ron\'s head. The moment his eyes fell upon Ron, all other concerns fled Harry\'s mind, for blood drenched the whole of Ron\'s left side and his face stood out, grayish-white, against the leaf-strewn earth. The Polyjuice Potion was wearing off now: Ron was halfway between Cattermole and himself in appearance, his hair turning redder and redder as his face drained of the little color it had left. "What\'s happened to him?" "Splinched," said Hermione, her fingers already busy at Ron\'s sleeve, where the blood was wettest and darkest. Harry watched, horrified, as she tore open Ron\'s shirt. He had always thought of Splinching as something comical, but this ... His insides crawled unpleasantly as Hermione laid bare Ron\'s upper arm, where a great chunk of flesh was missing, scooped cleanly away as though by a knife. "Harry, quickly, in my bag, there\'s a small bottle labeled \'Essence of Dittany\' -"\n"Bag - right -"\nHarry sped to the place where Hermione had landed, seized the tiny beaded bag, and thrust his hand inside it. At once, object after object began presenting itself to his touch: He felt the leather spines of books, woolly sleeves of jumpers, heels of shoes -\n"Quickly!" He grabbed his wand from the ground and pointed it into the depths of the magical bag. "Accio Dittany!" A small brown bottle zoomed out of the bag; he caught it and hastened back to Hermione and Ron, whose eyes were now half-closed, strips of white eyeball all that were visible between his lids. "He\'s fainted," said Hermione, who was also rather pale; she no longer looked like Mafalda, though her hair was still gray in places. "Unstopper it for me, Harry, my hands are shaking." Harry wrenched the stopper off the little bottle, Hermione took it and poured three drops of the potion onto the bleeding wound. Greenish smoke billowed upward and when it had cleared, Harry saw that the bleeding had stopped. The wound now looked several days old; new skin stretched over what had just been open flesh. "Wow," said Harry. "It\'s all I feel safe doing," said Hermione shakily. "There are spells that would put him completely right, but I daren\'t try in case I do them wrong and cause more damage. ... He\'s lost so much blood already. ..."\n"How did he get hurt? I mean" - Harry shook his head, trying to clear it, to make sense of whatever had just taken place - "why are we here?'}, {'chunk_id': 7714, 'title_num': 7, 'title': 'Harry Potter and the Deathly Hallows', 'chapter_num': 5, 'chapter_name': 'Fallen Warrior', 'passage': 'roared Hagrid. "I\'m sorry, Harry, but I had to check," said Lupin tersely. "We\'ve been betrayed. Voldemort knew that you were being moved tonight and the only people who could have told him were directly involved in the plan. You might have been an impostor." "So why aren\' you checkin\' me?" panted Hagrid, still struggling to fit through the door. "You\'re half-giant," said Lupin, looking up at Hagrid. "The Polyjuice Potion is designed for human use only."'}]
# print(KG_on_the_fly(quries))
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
