from copy import deepcopy
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput
import json
import sys
import numpy as np
import evaluate
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_metric
from autocorrect import Speller
from nltk.corpus import stopwords
import re
import emoji
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoImageProcessor, ViTModel, LlamaForSequenceClassification
import torch
from torch import nn
from transformers import DataCollatorWithPadding
from modeling_llama import LlamaForSequenceClassification
_CONFIG_FOR_DOC = "LlamaConfig"
disaster_types = ["Earthquake", "Flood", "Hurricane", "Wildfire"]

def get_disaster(event_name, label_text):
    if label_text == "not_informative":
        return "Non-informative"
    for disaster in disaster_types:
        if disaster.lower() in event_name.lower():
            return disaster
    return "Non-informative"

def load_and_process_tsv(file_path, images_dir):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['label_text'] == df['label_image']]
    df = df.drop_duplicates(subset=['tweet_id'])
    df = df[['tweet_text', 'image', 'event_name', 'label_text']].rename(columns={'tweet_text': 'text'})
    df['image_name'] = df['image'].apply(lambda x: os.path.basename(x))
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(images_dir, x))
    df['disaster'] = df.apply(lambda row: get_disaster(row['event_name'], row['label_text']), axis=1)
    disaster_labels = {"Non-informative": 0, "Earthquake": 1, "Flood": 2, "Hurricane": 3, "Wildfire": 4}
    df['label'] = df['disaster'].map(disaster_labels)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
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
    return ' '.join([abbreviations.get(w, w) for w in words])

def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "lxml").get_text()
    text = replace_emojis(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'rt\s+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = replace_abbreviations(text)
    text = spell(text)
    return re.sub(r'\s+', ' ', text).strip()

max_length = 128
model_id = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)
id2label = {0: "not_informative", 1: "informative"}
label2id = {v: k for k, v in id2label.items()}
tokenizer.pad_token = tokenizer.eos_token
text_model = LlamaForSequenceClassification.from_pretrained(model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id).bfloat16()
lora_r = 1000
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=600, lora_dropout=0.1)
text_model = get_peft_model(text_model, peft_config)
text_model.print_trainable_parameters()
text_model.config.pad_token_id = text_model.config.eos_token_id

def preprocess_text(examples):
    cleaned = [clean_text(t) for t in examples['text']]
    return tokenizer(cleaned, padding='longest', max_length=max_length, truncation=True)

dataset_dict = dataset_dict.map(preprocess_text, batched=True)
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess_image(examples):
    try:
        img = Image.open(examples['image_path']).convert('RGB')
    except:
        img = Image.new('RGB', (224,224))
    proc = vit_processor(img, return_tensors='pt')
    examples['pixel_values'] = proc['pixel_values'][0]
    return examples

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

image_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
text_model.config.output_hidden_states = True

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

# Hyperparameters for fusion
fusion_hidden_dim = 1024
num_labels = 2  # not_informative, informative

multimodal_model = MultiModalClassifier(
    text_model=text_model,
    image_model=image_model,
    fusion_hidden_dim=fusion_hidden_dim,
    num_labels=num_labels
)

def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)
    return {
        'precision': load_metric('precision').compute(predictions=preds, references=labels, average='weighted')['precision'],
        'recall': load_metric('recall').compute(predictions=preds, references=labels, average='weighted')['recall'],
        'f1': load_metric('f1').compute(predictions=preds, references=labels, average='weighted')['f1'],
        'accuracy': load_metric('accuracy').compute(predictions=preds, references=labels)['accuracy']
    }

args = TrainingArguments(output_dir="multimodal", learning_rate=7e-6, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=4, weight_decay=0.01, evaluation_strategy="epoch", save_strategy="epoch")
trainer = Trainer(model=multimodal_model, args=args, train_dataset=dataset_dict['train'], eval_dataset=dataset_dict['validation'], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
metrics = trainer.evaluate(dataset_dict['test'])
trainer.log_metrics("eval", metrics)
