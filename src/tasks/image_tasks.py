# src/tasks/run_image_tasks.py
from src.data.base_dataloader import load_data, preprocess_images_resnet, preprocess_images_vit
from src.models.base_models import load_image_model_for_classification
from src.utils.metrics import compute_metrics_classification
from transformers import TrainingArguments, Trainer, default_data_collator, ResNetForImageClassification, ViTForImageClassification

def run_task(task_config, model_config, training_config):
    # Load data
    dataset_dict = load_data(task_config["name"])
    
    # Determine model and preprocessing function
    image_model_type = task_config['image_model']
    if image_model_type == 'resnet':
        preprocess_func = preprocess_images_resnet
        model_class = ResNetForImageClassification
        model_id = model_config['resnet']['model_id']
    elif image_model_type == 'vit':
        preprocess_func = preprocess_images_vit
        model_class = ViTForImageClassification
        model_id = model_config['vit']['model_id']
    else:
        raise ValueError(f"Unsupported image model type: {image_model_type}")

    # Preprocess images
    processed_ds = dataset_dict.map(preprocess_func, batched=True, remove_columns=['image_path'])
    processed_ds.set_format(type='torch', columns=['pixel_values', 'labels'])

    # Load model
    num_labels = len(task_config["labels"])
    model = model_class.from_pretrained(
        model_id, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # Set up and run trainer
    training_args = TrainingArguments(**training_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_ds["train"],
        eval_dataset=processed_ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_classification,
    )

    trainer.train()
    metrics = trainer.evaluate(processed_ds["test"])
    trainer.log_metrics("eval", metrics)
    print("Test metrics:", metrics)