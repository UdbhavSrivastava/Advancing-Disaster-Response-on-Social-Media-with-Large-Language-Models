
import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_metric
from PIL import Image
from torchvision import transforms
import torch
from transformers import ResNetConfig, ResNetForImageClassification, TrainingArguments, Trainer, default_data_collator

def load_and_process_tsv(file_path, images_dir):
    df = pd.read_csv(file_path, sep='\t')
    # keep only the image path and its label
    df = df[['image', 'label_image']]
    # map label_image → numeric
    df['label'] = df['label_image'].map({
        'not_informative': 0,
        'informative':     1
    })
    # extract filename + rebuild full path
    df['image_name'] = df['image'].apply(lambda x: os.path.basename(x))
    df['image_path'] = df['image_name'].apply(lambda fn: os.path.join(images_dir, fn))
    return Dataset.from_pandas(df, preserve_index=False)

data_dir   = "/scratch/user/udbhav23/CrisisMMD_v2.0/CrisisMMD_v2.0/Informative/"
images_dir = "/scratch/user/udbhav23/CrisisMMD_v2.0/CrisisMMD_v2.0/images"
files = {
    "train":      "task_informative_text_img_train.tsv",
    "validation": "task_informative_text_img_dev.tsv",
    "test":       "task_informative_text_img_test.tsv",
}

dataset_dict = DatasetDict({
    split: load_and_process_tsv(os.path.join(data_dir, fname), images_dir)
    for split, fname in files.items()
})

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_images(examples):
    pixels = [image_transform(Image.open(path).convert("RGB")) for path in examples["image_path"]]
    return {"pixel_values": pixels, "labels": examples["label"]}

processed_ds = dataset_dict.map(preprocess_images, batched=True, remove_columns=["text", "image", "event_name", "label_image", "image_name", "image_path", "disaster"])
processed_ds.set_format(type="torch", columns=["pixel_values", "labels"])

metric_acc = load_metric("accuracy")
metric_prec = load_metric("precision")
metric_rec = load_metric("recall")
metric_f1 = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "precision": metric_prec.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": metric_rec.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# =============================================================================
# 5. Compute Metrics
# =============================================================================
metric_acc = load_metric("accuracy")
metric_prec = load_metric("precision")
metric_rec = load_metric("recall")
metric_f1 = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "precision": metric_prec.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall":    metric_rec.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1":        metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

num_labels = 2  

config = ResNetConfig.from_pretrained("microsoft/resnet-101")
config.num_labels = num_labels
model = ResNetForImageClassification.from_pretrained(
    "microsoft/resnet-101",
    config=config,
    ignore_mismatched_sizes=True  
)


training_args = TrainingArguments(output_dir="resnet101_disaster_type5", per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=10, learning_rate=9e-4, weight_decay=0.01, evaluation_strategy="epoch", save_strategy="epoch", logging_strategy="epoch")

trainer = Trainer(model=model, args=training_args, train_dataset=processed_ds["train"], eval_dataset=processed_ds["validation"], data_collator=default_data_collator, compute_metrics=compute_metrics)

trainer.train()

test_metrics = trainer.evaluate(processed_ds["test"])
trainer.log_metrics("test", test_metrics)
print("Test metrics:", test_metrics)
