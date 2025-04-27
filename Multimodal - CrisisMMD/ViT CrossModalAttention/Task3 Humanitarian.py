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

import os
import pandas as pd
from datasets import Dataset, DatasetDict

data_dir = ""
images_dir = ""
files = {
    "train": "task_humanitarian_text_img_train.tsv",
    "validation": "task_humanitarian_text_img_dev.tsv",
    "test": "task_humanitarian_text_img_test.tsv",
}

allowed_labels = {
    "not_humanitarian": 0,
    "affected_individuals": 1,
    "infrastructure_and_utility_damage": 2,
    "other_relevant_information": 3,
    "rescue_volunteering_or_donation_effort": 4,
}

def load_and_clean(file_path, images_dir):
    df = pd.read_csv(file_path, sep="\t")
    df = df[["image", "label_image"]]
    df = df[df["label_image"].isin(allowed_labels)]
    df["label"] = df["label_image"].map(allowed_labels)
    df["image_name"] = df["image"].apply(lambda x: os.path.basename(x))
    df["image_path"] = df["image_name"].apply(lambda fn: os.path.join(images_dir, fn))
    return Dataset.from_pandas(df, preserve_index=False)

dataset_dict = DatasetDict({
    split: load_and_clean(os.path.join(data_dir, fname), images_dir)
    for split, fname in files.items()
})

dataset_dict = DatasetDict({split: load_and_clean(os.path.join(data_dir, f), images_dir) for split, f in files.items()})

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
    def __init__(self, tokenizer): self.text_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    def __call__(self, features):
        imgs = [f.pop('pixel_values') for f in features]
        imgs = [i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in imgs]
        batch = self.text_collator(features)
        batch['pixel_values'] = torch.stack(imgs)
        return batch

data_collator = CustomDataCollator(tokenizer)

class CrossModalFusion(nn.Module):
    def __init__(self, t_dim, i_dim, fusion_dim, num_heads=8):
        super().__init__()
        self.text_proj = nn.Linear(t_dim, fusion_dim)
        self.image_proj = nn.Linear(i_dim, fusion_dim)
        self.cross_attn_text = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        self.cross_attn_image = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        self.ln_text = nn.LayerNorm(fusion_dim)
        self.ln_image = nn.LayerNorm(fusion_dim)
    def forward(self, t, i):
        t_p = self.text_proj(t)
        i_p = self.image_proj(i)
        att_t, _ = self.cross_attn_text(t_p, i_p, i_p)
        t_f = self.ln_text(t_p + att_t)
        att_i, _ = self.cross_attn_image(i_p, t_p, t_p)
        i_f = self.ln_image(i_p + att_i)
        t_pool = t_f.mean(1)
        i_pool = i_f.mean(1)
        return (t_pool + i_pool) / 2

class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, img_model, fusion_dim, num_labels, heads=8):
        super().__init__()
        self.text_model = text_model
        self.image_model = img_model
        t_dim = text_model.config.hidden_size
        i_dim = img_model.config.hidden_size if hasattr(img_model.config, 'hidden_size') else 768
        self.fusion = CrossModalFusion(t_dim, i_dim, fusion_dim, heads)
        self.classifier = nn.Linear(fusion_dim, num_labels)
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        t_out = self.text_model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        t_hidden = t_out.hidden_states[-1].float()
        i_hidden = self.image_model(pixel_values=pixel_values, return_dict=True).last_hidden_state.float()
        fused = self.fusion(t_hidden, i_hidden)
        logits = self.classifier(fused)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return dict(loss=loss, logits=logits) if loss is not None else dict(logits=logits)

fusion_dim = 1024
num_labels = 5
image_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
text_model.config.output_hidden_states = True
model = MultiModalClassifier(text_model, image_model, fusion_dim, num_labels)

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
trainer = Trainer(model=model, args=args, train_dataset=dataset_dict['train'], eval_dataset=dataset_dict['validation'], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
metrics = trainer.evaluate(dataset_dict['test'])
trainer.log_metrics("eval", metrics)
