# src/tasks/run_multimodal_tasks.py
from src.data.base_dataloader import load_data, preprocess_text, MultimodalDataCollator, preprocess_images_resnet, preprocess_images_vit
from src.models.base_models import load_llama_model, load_image_model
from src.models.fusion_models import MultimodalConcatenation, CrossModalFusionAttention
from src.utils.metrics import compute_metrics_classification
from transformers import TrainingArguments, Trainer

def run_task(task_config, model_config, training_config):
    # Load data
    dataset_dict = load_data(task_config["name"])

    # Load base models
    tokenizer, text_model = load_llama_model(model_config['llama'], len(task_config['labels']), task_type='sequence_classification')
    image_model = load_image_model(task_config['image_model'])
    
    # Preprocess data
    tokenized_ds = dataset_dict.map(
        lambda examples: preprocess_text(examples, tokenizer, model_config['llama']['max_length']),
        batched=True
    ).map(
        preprocess_images_resnet if task_config['image_model'] == 'resnet' else preprocess_images_vit,
        batched=True
    )
    tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values', 'labels'])

    # Set up fusion model
    fusion_type = task_config['fusion']
    if fusion_type == 'concat':
        fusion_model = MultimodalConcatenation(
            text_model, image_model, model_config['fusion_models']['multimodal_concat']['fusion_hidden_dim'], 
            len(task_config['labels'])
        )
    elif fusion_type == 'cross_attention':
        fusion_model = CrossModalFusionAttention(
            text_model, image_model, model_config['fusion_models']['multimodal_cross_attention']['fusion_dim'],
            len(task_config['labels']), num_heads=model_config['fusion_models']['multimodal_cross_attention']['num_heads']
        )
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")

    # Set up and run trainer
    data_collator = MultimodalDataCollator(tokenizer)
    training_args = TrainingArguments(**training_config)

    trainer = Trainer(
        model=fusion_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_ds["test"])
    trainer.log_metrics("eval", metrics)
    print("Test metrics:", metrics)