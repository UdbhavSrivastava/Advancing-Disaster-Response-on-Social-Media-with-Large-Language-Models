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
from transformers import ResNetModel
import torch.nn.functional as F
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

class MultimodalFusionModel(nn.Module):
    def __init__(self, text_model, num_labels, fusion_hidden_size=512):
        super().__init__()
        self.text_model = text_model  # Your LLaMA-based text model
        # Load ResNet-101 model as image encoder
        self.image_model = ResNetModel.from_pretrained("microsoft/resnet-101")
        
        # Retrieve hidden sizes
        text_hidden_size = self.text_model.config.hidden_size
        image_hidden_size = self.image_model.config.hidden_size if hasattr(self.image_model.config, "hidden_size") else 2048
        
        fusion_input_size = text_hidden_size + image_hidden_size
        # Simple classifier head on top of fused representation
        self.classifier = nn.Linear(fusion_input_size, num_labels)
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
    # Process text input
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    # Mean pooling over the last hidden state for text features
        text_features = text_outputs.hidden_states[-1].mean(dim=1)  # shape: (batch, text_hidden_size)
    
    # Process image input
        image_outputs = self.image_model(pixel_values)
        if image_outputs.last_hidden_state.dim() == 4:
            image_features = F.adaptive_avg_pool2d(image_outputs.last_hidden_state, (1, 1))
            image_features = image_features.view(image_features.size(0), -1)  # shape: (batch, image_hidden_size)
        else:
            image_features = image_outputs.last_hidden_state
    
    # Fuse features via concatenation
        fused_features = torch.cat([text_features, image_features], dim=1)
        logits = self.classifier(fused_features)
    
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
    
        return {"loss": loss, "logits": logits}

num_labels = len(label2id)
fusion_model = MultimodalFusionModel(text_model=text_model, num_labels=num_labels)

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
trainer = Trainer(model=fusion_model, args=training_args, train_dataset=dataset_dict["train"], eval_dataset=dataset_dict["validation"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
metrics = trainer.evaluate(dataset_dict["test"])
trainer.log_metrics("eval", metrics)
