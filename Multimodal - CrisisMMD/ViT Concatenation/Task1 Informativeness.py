from copy import deepcopy
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput
import json
import sys
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_metric
import os
import pandas as pd
from autocorrect import Speller
from nltk.corpus import stopwords
import re
import emoji
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import torch
from torch import nn
from modeling_llama import LlamaForSequenceClassification
from transformers import DataCollatorWithPadding
_CONFIG_FOR_DOC = "LlamaConfig"

def load_and_process_tsv(file_path, images_dir):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['label_text'] == df['label_image']]
    df = df.drop_duplicates(subset=['tweet_id'])
    df = df[['tweet_text', 'image', 'label_text']].rename(columns={'tweet_text': 'text', 'label_text': 'label_text'})
    df['label'] = df['label_text'].map({'not_informative': 0, 'informative': 1})
    df['image_name'] = df['image'].apply(lambda x: os.path.basename(x))
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(images_dir, x))
    return Dataset.from_pandas(df, preserve_index=False)

data_dir = ""
images_dir = ""
files = {"train": "task_informative_text_img_train.tsv", "validation": "task_informative_text_img_dev.tsv", "test": "task_informative_text_img_test.tsv"}
dataset_dict = DatasetDict({
    split: load_and_process_tsv(os.path.join(data_dir, filename), images_dir)
    for split, filename in files.items()
})

stop_words = set(stopwords.words('english'))
spell = Speller(lang='en')
abbreviations = {"thnx": "thanks", "thx": "thanks", "btw": "by the way", "pls": "please", "plz": "please", "u": "you", "r": "are", "ur": "your", "y": "why", "b4": "before", "gr8": "great", "imo": "in my opinion", "idk": "I don't know", "w8": "wait", "bday": "birthday"}

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def replace_abbreviations(text):
    words = text.split()
    return ' '.join([abbreviations[word] if word in abbreviations else word for word in words])

def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "lxml").get_text()
    text = replace_emojis(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'rt\s+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = replace_abbreviations(text)
    text = spell(text)
    words = text.split()
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

max_length = 128
model_id = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)
id2label = {0: "not_informative", 1: "informative"}
label2id = {v: k for k, v in id2label.items()}
tokenizer.pad_token = tokenizer.eos_token
text_model = LlamaForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).bfloat16()
lora_r = 1000
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=600, lora_dropout=0.1)
text_model = get_peft_model(text_model, peft_config)
text_model.print_trainable_parameters()
text_model.config.pad_token_id = text_model.config.eos_token_id

def preprocess_text_function(examples):
    cleaned = [clean_text(t) for t in examples['text']]
    return tokenizer(cleaned, padding='longest', max_length=max_length, truncation=True)

dataset_dict = dataset_dict.map(preprocess_text_function, batched=True)
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess_image(example):
    try:
        image = Image.open(example["image_path"]).convert("RGB")
    except:
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))
    processed = vit_processor(image, return_tensors="pt")
    example["pixel_values"] = processed["pixel_values"][0]
    return example

dataset_dict = dataset_dict.map(preprocess_image)

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.text_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features):
        # Extract and remove image features from the features dict
        image_features = [feat.pop("pixel_values") for feat in features]
        # Ensure each image feature is a tensor (convert if necessary)
        image_features = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x)
            for x in image_features
        ]
        batch = self.text_collator(features)
        # Stack image pixel values into a tensor
        batch["pixel_values"] = torch.stack(image_features)
        return batch



data_collator = CustomDataCollator(tokenizer)

class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, fusion_hidden_dim, num_labels):
        super().__init__()
        self.text_model = text_model  # LLaMA text model with PEFT applied
        self.image_model = image_model  # ViT image model
        
        # Get dimensions from both modalities
        text_hidden_dim = text_model.config.hidden_size
        image_hidden_dim = image_model.config.hidden_size if hasattr(image_model.config, "hidden_size") else 768
        
        # Fusion layer to combine text and image embeddings
        self.fusion_proj = nn.Linear(text_hidden_dim + image_hidden_dim, fusion_hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(fusion_hidden_dim, num_labels)
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Process text through LLaMA (using underlying model to obtain embeddings)
        text_outputs = self.text_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,  # Ensure hidden states are returned
        )
        # Mean pooling of the text hidden states
        text_hidden = text_outputs.hidden_states[-1]
        text_embeds = text_hidden.mean(dim=1)
        # Process image through ViT
        image_outputs = self.image_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        # Use the pooled output from ViT (CLS token representation)
        image_embeds = image_outputs.pooler_output
        
        # Concatenate text and image features
        combined = torch.cat([text_embeds, image_embeds], dim=1)
        fused = self.activation(self.fusion_proj(combined))
        logits = self.classifier(fused)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

fusion_hidden_dim = 1024
num_labels = 2
image_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
text_model.config.output_hidden_states = True
multimodal_model = MultiModalClassifier(text_model=text_model, image_model=image_model, fusion_hidden_dim=fusion_hidden_dim, num_labels=num_labels, fusion_dim=fusion_hidden_dim, num_heads=8)

def compute_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = metric1.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

epochs = 4
batch_size = 16
learning_rate = 7e-6
training_args = TrainingArguments(output_dir="multimodal_informative_viT_cross", learning_rate=learning_rate, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=epochs, weight_decay=0.01, evaluation_strategy="epoch", save_strategy="epoch")
trainer = Trainer(model=multimodal_model, args=training_args, train_dataset=dataset_dict["train"], eval_dataset=dataset_dict["validation"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
metrics = trainer.evaluate(dataset_dict["test"])
trainer.log_metrics("eval", metrics)
