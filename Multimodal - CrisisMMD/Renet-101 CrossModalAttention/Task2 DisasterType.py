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
from transformers import ResNetModel
import torch.nn.functional as F
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

from PIL import Image
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image)
def multimodal_preprocess(examples):
    # Tokenize text
    tokenized = tokenizer(examples["text"], padding="longest", max_length=max_length, truncation=True)
    # Process images: load image tensor from image_path for each example
    tokenized["pixel_values"] = [load_image(path) for path in examples["image_path"]]
    # Add labels if present
    if "label" in examples:
        tokenized["labels"] = examples["label"]
    return tokenized

# Map the multimodal preprocessing to all splits
tokenized_ds = dataset_dict.map(multimodal_preprocess, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

class CrossModalFusionAttention(nn.Module):
    def __init__(self, text_model, num_labels, fusion_dim=512, num_heads=8):
        super().__init__()
        self.text_model = text_model  # Pre-trained LLaMA-based text model
        
        # Load ResNet-101 as the image encoder
        self.image_model = ResNetModel.from_pretrained("microsoft/resnet-101")
        
        # Projection layers to map text and image features into a common fusion space.
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, fusion_dim)
        # For ResNet-101, we assume the output feature dimension is 2048.
        self.image_proj = nn.Linear(2048, fusion_dim)
        
        # Multi-head cross-attention: text tokens (queries) attend to image features (keys/values).
        # Note: batch_first=True means input/output tensors are of shape (B, seq_len, fusion_dim).
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        
        # Final classifier that takes the fused representation and outputs logits.
        self.classifier = nn.Linear(fusion_dim, num_labels)
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Process text: Obtain all token embeddings from the text model.
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # Extract the last hidden state: shape (B, seq_len, d_text)
        text_hidden = text_outputs.hidden_states[-1]
        # Cast text hidden states to float32 to match projection weights
        text_hidden = text_hidden.float()
        # Project text features to the fusion dimension.
        text_proj = self.text_proj(text_hidden)  # (B, seq_len, fusion_dim)
        
        # Process image input using ResNet-101.
        image_outputs = self.image_model(pixel_values)
        # ResNet-101 typically returns a 4D tensor in last_hidden_state: (B, C, H, W)
        if image_outputs.last_hidden_state.dim() == 4:
            B, C, H, W = image_outputs.last_hidden_state.shape
            # Reshape to a sequence: (B, H*W, C)
            image_seq = image_outputs.last_hidden_state.view(B, C, H * W).transpose(1, 2)
        else:
            image_seq = image_outputs.last_hidden_state  # Assume (B, seq_len_img, C)
        
        # Project image features to the fusion dimension.
        image_proj = self.image_proj(image_seq)  # (B, seq_len_img, fusion_dim)
        
        # Cross-modal attention: let text queries attend to image features.
        attn_output, _ = self.cross_attn(query=text_proj, key=image_proj, value=image_proj)
        # Add a residual connection to preserve original text information.
        fused_text = text_proj + attn_output  # (B, seq_len, fusion_dim)
        # Pool over the token dimension to get a single representation per tweet.
        fused_representation = fused_text.mean(dim=1)  # (B, fusion_dim)
        
        logits = self.classifier(fused_representation)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# Corrected instantiation: using CrossModalFusionAttention instead of MultimodalFusionModel.
num_labels = len(label2id)
fusion_model = CrossModalFusionAttention(text_model=text_model, num_labels=num_labels)


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
trainer = Trainer(model=fusion_model, args=args, train_dataset=dataset_dict['train'], eval_dataset=dataset_dict['validation'], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
metrics = trainer.evaluate(dataset_dict['test'])
trainer.log_metrics("eval", metrics)
