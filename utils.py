# Description: This file contains utility functions that are used in the main script.

from transformers import AutoTokenizer, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset

class NLPBase:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.trainer = None

    def preprocess_function(self, examples):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class NER(NLPBase):
    def __init__(self, model_name, label_list):
        super().__init__(model_name)
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
        
        

    def preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    # print(label)
                    # print(label[word_idx])
                    label_ids.append(label[word_idx])
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self, dataset, training_args, model):
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        model = model.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,

        )
        self.trainer.train()
    def eval(self):
        """
        Evaluate the model on the validation set.
        """
        if self.trainer is None:
            raise ValueError("Model has not been trained yet.")
        
        results = self.trainer.evaluate()
        print(f"Evaluation results: {results}")
        return results



class QA(NLPBase):
    def __init__(self, model_name):
        super().__init__(model_name)

    """
    Assume the dataset has the following structure:
    {
    "context": "Elizabeth Bennett met Mr. Darcy at a ball in the town of Meryton...",
    "question": "Where did Elizabeth Bennett meet Mr. Darcy?",
    "answers": [
        {
        "text": "at a ball in the town of Meryton",
        "answer_start": 31
        }
    ]
    }
    """
    def preprocess_function(self, examples):
        # Truncate or pad to max_length, handle offsets for extractive QA
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # TODO: Handle answers according to real our data structure

        return inputs

    def train(self, dataset, training_args, model):
        processed_dataset = dataset.map(self.preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
        model = model.from_pretrained(self.model_name)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"]
        )
        trainer.train()