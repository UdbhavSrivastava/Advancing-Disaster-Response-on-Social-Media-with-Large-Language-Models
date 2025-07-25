from copy import deepcopy

from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput


import re
import string
import emoji
from bs4 import BeautifulSoup
from word2number import w2n
from autocorrect import Speller
from nltk.corpus import stopwords

import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

_CONFIG_FOR_DOC = "LlamaConfig"
from modeling_llama import LlamaForSequenceClassification
import pandas as pd
from datasets import Dataset, DatasetDict

# Define the file paths
data_dir = "/"
files = {
    "train": "task_humanitarian_train.tsv",
    "validation": "task_humanitarian_dev.tsv",
    "test": "task_humanitarian_test.tsv",
}

# Load all datasets and merge
df_list = [pd.read_csv(file, sep="\t") for file in files.values()]
merged_df = pd.concat(df_list, ignore_index=True)

# Filter only "hurricane_maria"
filtered_df = merged_df[merged_df["event_name"] == "event_name"]

# Map labels
label_mapping = {"damage": 1, "no_damage": 0}
filtered_df["label"] = filtered_df["damage"].map(label_mapping)
# 🔹 Shuffle dataset before splitting
filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Stratified 80-20 split for train-test
train_df, test_df = train_test_split(
    filtered_df, test_size=0.08, stratify=filtered_df["label"], random_state=42
)

# Further split train into 90% train, 10% validation
train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

# Convert to Hugging Face Dataset
def convert_to_hf_dataset(df):
    return Dataset.from_pandas(df[["tweet_text", "label"]].rename(columns={"tweet_text": "text"}), preserve_index=False)

dataset_dict = DatasetDict({
    "train": convert_to_hf_dataset(train_df),
    "validation": convert_to_hf_dataset(val_df),
    "test": convert_to_hf_dataset(test_df),
})
# Print the structure of the DatasetDict
print(dataset_dict)


# Load English stopwords
stop_words = set(stopwords.words('english'))

# Spelling corrector
spell = Speller(lang='en')

# Common abbreviations dictionary
abbreviations = {
    "thnx": "thanks", "thx": "thanks", "btw": "by the way", "pls": "please", "plz": "please",
    "u": "you", "r": "are", "ur": "your", "y": "why", "b4": "before", "gr8": "great",
    "imo": "in my opinion", "idk": "I don't know", "w8": "wait", "bday": "birthday"
}

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))  # Converts emojis to text

def replace_abbreviations(text):
    words = text.split()
    return ' '.join([abbreviations[word] if word in abbreviations else word for word in words])

def clean_text(text):
    text = text.lower()  # Lowercasing
    
    text = BeautifulSoup(text, "lxml").get_text()  # Remove HTML tags
    text = replace_emojis(text)  # Convert emojis to text
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'rt\s+', '', text)  # Remove retweet (RT)
    #text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    #text = re.sub(r'\d+', '', text)  # Remove numbers
    #text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation
    
    text = replace_abbreviations(text)  # Replace abbreviations
    text = spell(text)  # Fix misspelled words
    
    words = text.split()
    #words = [word for word in words if word not in stop_words]  # Remove stopwords (for TF-IDF & Word2Vec)
    
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    return text

for split in dataset_dict:
    dataset_dict[split] = dataset_dict[split].map(lambda x: {"text": clean_text(x["text"])})
    
print(dataset_dict["train"][0])

epochs = 4
batch_size = 16
learning_rate = 7e-6
lora_r = 2500
max_length = 128
model_id = 'meta-llama/Llama-2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_id)
id2label = { 0:"no_damage",1:"damage" }
label2id = {v: k for k, v in id2label.items()}
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForSequenceClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
).bfloat16()

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=600, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.pad_token_id = model.config.eos_token_id

from datasets import load_metric
import numpy as np

def compute_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels,
                                average="weighted")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels,
                             average="weighted")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels,
                         average="weighted")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)[
        "accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy}

test_name = 'test'
text_name = 'text'

def preprocess_function(examples):
    global text_name
    if isinstance(text_name, str):
        d = examples[text_name]
    else:
        d = examples[text_name[0]]
        for n in text_name[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t

    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)

tokenized_ds = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

training_args = TrainingArguments(
    output_dir="output",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #load_best_model_at_end=True,
    #push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate(tokenized_ds['test'])
trainer.log_metrics("eval", metrics)