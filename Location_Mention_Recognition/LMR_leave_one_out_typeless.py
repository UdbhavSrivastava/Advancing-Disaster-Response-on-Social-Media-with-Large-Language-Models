from copy import deepcopy

from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput
import os
import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from evaluate import load
from modeling_llama import UnmaskingLlamaForTokenClassification

_CONFIG_FOR_DOC = "LlamaConfig"


def read_bilou_file(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as file:
        sentence = []
        label = []
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                word, tag = line.strip().split()
                sentence.append(word)
                label.append(tag)
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

def read__data(base_path):
    exclude_folder = 'enter your testing event name'
    train_sentences, train_labels = [], []
    dev_sentences, dev_labels = [], []
    test_sentences, test_labels = [], []

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) and not folder_name.startswith('.'):
            train_file = os.path.join(folder_path, 'train.txt')
            dev_file = os.path.join(folder_path, 'dev.txt')
            test_file = os.path.join(folder_path, 'test.txt')

            # Check if this is the canada_wildfires_2016 folder
            if folder_name == exclude_folder:
                # For test data
                if os.path.exists(train_file):
                    dev_sents, dev_lbls = read_bilou_file(train_file)
                    dev_sentences.extend(dev_sents)
                    dev_labels.extend(dev_lbls)
                if os.path.exists(dev_file):
                    dev_sents, dev_lbls = read_bilou_file(dev_file)
                    dev_sentences.extend(dev_sents)
                    dev_labels.extend(dev_lbls)
                if os.path.exists(test_file):
                    test_sents, test_lbls = read_bilou_file(test_file)
                    test_sentences.extend(test_sents)
                    test_labels.extend(test_lbls)
            else:
                # For other folders
                if os.path.exists(train_file):
                    train_sents, train_lbls = read_bilou_file(train_file)
                    train_sentences.extend(train_sents)
                    train_labels.extend(train_lbls)
                if os.path.exists(dev_file):
                    dev_sents, dev_lbls = read_bilou_file(dev_file)
                    dev_sentences.extend(dev_sents)
                    dev_labels.extend(dev_lbls)
                if os.path.exists(test_file):
                    dev_sents, dev_lbls = read_bilou_file(test_file)
                    dev_sentences.extend(dev_sents)
                    dev_labels.extend(dev_lbls)

    return train_sentences, train_labels, dev_sentences, dev_labels, test_sentences, test_labels

base_path = '/path/to/your/dataset'
train_sentences, train_labels, dev_sentences, dev_labels, test_sentences, test_labels = read__data(base_path)


def bilou_to_spans(tokens, labels):
    spans = []
    current_entity = []

    for token, label in zip(tokens, labels):
        if label.startswith('B-') or label.startswith('U-'):
            if current_entity:
                spans.append(' '.join(current_entity))
                current_entity = []
            current_entity.append(f"{label[2:]}: {token}")
        elif label.startswith('I-') or label.startswith('L-'):
            if current_entity:
                current_entity.append(f"{label[2:]}: {token}")
        elif label == 'O':
            if current_entity:
                spans.append(' '.join(current_entity))
                current_entity = []

    if current_entity:
        spans.append(' '.join(current_entity))

    return spans

def process_dataset(sentences, labels):
    data = []
    for sentence, label in zip(sentences, labels):
        spans = bilou_to_spans(sentence, label)
        data.append({
            'tokens': sentence,
            'ner_tags': [label_map[tag] for tag in label],  
            'langs': ['en'] * len(sentence),
            'spans': spans
        })
    return data

label_map = {  
    'O': 0,
    'B-LOC': 1,
    'L-LOC': 2,
    'I-LOC': 3,
    'U-LOC': 4
}

train_data = process_dataset(train_sentences, train_labels)
dev_data = process_dataset(dev_sentences, dev_labels)
test_data = process_dataset(test_sentences, test_labels)

from datasets import Dataset, DatasetDict

def convert_to_dict(data):
    # Initialize a dictionary with empty lists for each key
    dict_data = {key: [] for key in data[0].keys()}
    
    # Populate the dictionary with values from each example in the list
    for example in data:
        for key, value in example.items():
            dict_data[key].append(value)
    
    return dict_data

# Convert the list of dictionaries to a dictionary of lists
train_dict = convert_to_dict(train_data)
dev_dict = convert_to_dict(dev_data)
test_dict = convert_to_dict(test_data)

# Convert the dictionaries to Dataset objects
train_dataset = Dataset.from_dict(train_dict)
dev_dataset = Dataset.from_dict(dev_dict)
test_dataset = Dataset.from_dict(test_dict)

# Create a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": test_dataset,
    "test": dev_dataset
})


def process_dataset(sentences, labels):
    data = []
    for sentence, label in zip(sentences, labels):
        data.append({
            'tokens': sentence,
            'ner_tags': [label2id[tag] for tag in label]
        })
    return data

label2id = { 'O': 0,
    'B-LOC': 1,
    'L-LOC': 2,
    'I-LOC': 3,
    'U-LOC': 4}
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

epochs = 15
batch_size = 8
learning_rate = 5e-4
max_length = 128
model_id = 'meta-llama/Llama-2-7b-hf'
lora_r = 12


tokenizer = AutoTokenizer.from_pretrained(model_id)
seqeval = evaluate.load("seqeval")

tokenizer.pad_token = tokenizer.eos_token

model = UnmaskingLlamaForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
).bfloat16()

peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=lora_r, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="output",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions, labels, _ = trainer.predict(tokenized_ds["test"])
predictions = np.argmax(predictions, axis=-1)

metric = load("seqeval")

true_labels = [
    [id2label[l] for l in label  if l != -100]
    for label in labels
]

true_predictions = [
    [id2label[p] for (p, l) in zip(prediction, label)  if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)