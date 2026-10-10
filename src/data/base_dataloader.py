import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, DataCollatorWithPadding


SPLIT_FILES = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}

DISASTER_LABELS = {
    "Non-informative": 0,
    "Earthquake": 1,
    "Flood": 2,
    "Hurricane": 3,
    "Wildfire": 4,
}

_VIT_PROCESSOR = None


def _require_existing_dir(path, name):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{name} does not exist: {path}. Pass an override, for example "
            f"`--{name} C:\\path\\to\\CrisisMMD_v2.0`."
        )
    return path


def _task_data_dir(task_config):
    base_paths = task_config.get("base_paths", {})
    data_root = _require_existing_dir(base_paths.get("data_dir", ""), "data_dir")
    data_path = task_config.get("data_path", "")
    candidate = data_root / data_path
    if candidate.exists():
        return candidate
    return data_root


def _split_paths(task_config):
    data_dir = _task_data_dir(task_config)
    prefix = task_config.get("file_prefix", "")
    return {
        split: data_dir / f"{prefix}{suffix}"
        for split, suffix in SPLIT_FILES.items()
    }


def _labels(task_config):
    return task_config["labels"]


def _label_source(row, prefer_image=False):
    if prefer_image and "label_image" in row and pd.notna(row["label_image"]):
        return row["label_image"]
    if "label_text" in row and pd.notna(row["label_text"]):
        return row["label_text"]
    if "label" in row and pd.notna(row["label"]):
        return row["label"]
    if "label_image" in row and pd.notna(row["label_image"]):
        return row["label_image"]
    return None


def _disaster_from_event(event_name, original_label):
    if original_label == "not_informative":
        return "Non-informative"

    event = str(event_name).lower()
    if "earthquake" in event:
        return "Earthquake"
    if "flood" in event:
        return "Flood"
    if "hurricane" in event:
        return "Hurricane"
    if "wildfire" in event or "fire" in event:
        return "Wildfire"
    return None


def _resolve_image_path(raw_image, images_dir):
    if pd.isna(raw_image):
        return ""

    raw = str(raw_image)
    raw_path = Path(raw)
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)

    images_root = Path(images_dir).expanduser()
    candidates = [
        images_root / raw,
        images_root / Path(raw).name,
        images_root / raw.replace("data_image/", "").replace("data_image\\", ""),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _prepare_frame(df, task_config):
    labels = _labels(task_config)
    template_name = task_config.get("template_name")
    prefer_image_label = task_config.get("type") == "image_only"

    if not prefer_image_label and "label_text" in df.columns and "label_image" in df.columns:
        df = df[df["label_text"] == df["label_image"]].copy()

    if "tweet_id" in df.columns:
        df = df.drop_duplicates(subset=["tweet_id"])

    out = pd.DataFrame()
    out["text"] = df["tweet_text"] if "tweet_text" in df.columns else ""

    if template_name == "disaster_type":
        label_text = df.apply(
            lambda row: _disaster_from_event(
                row.get("event_name", ""),
                _label_source(row, prefer_image=prefer_image_label),
            ),
            axis=1,
        )
    else:
        label_text = df.apply(lambda row: _label_source(row, prefer_image=prefer_image_label), axis=1)

    out["label_text"] = label_text
    out = out[out["label_text"].isin(labels)].copy()
    out["label"] = out["label_text"].map(labels).astype(int)

    if "image" in df.columns:
        images_dir = task_config.get("base_paths", {}).get("images_dir", "")
        out["image"] = df.loc[out.index, "image"]
        out["image_path"] = out["image"].apply(lambda value: _resolve_image_path(value, images_dir))

    return out.reset_index(drop=True)


def load_data(task_config):
    if task_config.get("type") == "token_classification":
        raise NotImplementedError(
            "LMR token-classification loading is not wired in the refactored runner yet. "
            "Run the existing LMR scripts directly or add an LMR-specific loader."
        )

    split_paths = _split_paths(task_config)
    datasets = {}
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split for {task_config['name']}: {path}")
        df = pd.read_csv(path, sep="\t")
        datasets[split] = Dataset.from_pandas(_prepare_frame(df, task_config), preserve_index=False)
    return DatasetDict(datasets)


def preprocess_text(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        padding="longest",
        max_length=max_length,
        truncation=True,
    )


def tokenize_and_align_labels(examples, tokenizer, label_map, max_length):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


_RESNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


def preprocess_images_resnet(examples):
    pixels = [_RESNET_TRANSFORM(_open_image(path)) for path in examples["image_path"]]
    return {"pixel_values": pixels, "labels": examples["label"]}


def preprocess_images_vit(examples):
    global _VIT_PROCESSOR
    if _VIT_PROCESSOR is None:
        _VIT_PROCESSOR = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    images = [_open_image(path) for path in examples["image_path"]]
    processed = _VIT_PROCESSOR(images, return_tensors="pt")
    return {"pixel_values": list(processed["pixel_values"]), "labels": examples["label"]}


class MultimodalDataCollator:
    def __init__(self, tokenizer):
        self.text_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features):
        pixel_values = [feature.pop("pixel_values") for feature in features]
        pixel_values = [
            value if isinstance(value, torch.Tensor) else torch.tensor(value)
            for value in pixel_values
        ]
        batch = self.text_collator(features)
        batch["pixel_values"] = torch.stack(pixel_values)
        return batch
