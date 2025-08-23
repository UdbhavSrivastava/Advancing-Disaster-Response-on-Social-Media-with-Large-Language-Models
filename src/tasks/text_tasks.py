# src/tasks/run_text_tasks.py
from src.data.base_dataloader import load_data, preprocess_text, tokenize_and_align_labels
from src.models.base_models import load_llama_model
from src.utils.metrics import compute_metrics_classification, compute_metrics_token_classification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForTokenClassification

def run_task(task_config, model_config, training_config):
    # Determine the task type and load appropriate model
    task_type = task_config.get("type", "classification")
    num_labels = len(task_config["labels"])

    if task_type == "token_classification":
        model_name = "llama_token_cls"
        tokenizer, model = load_llama_model(model_config[model_name], num_labels, task_type=task_type)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        compute_metrics = lambda p: compute_metrics_token_classification(p, list(task_config["labels"].keys()))
        preprocess_func = lambda examples: tokenize_and_align_labels(examples, tokenizer, task_config["labels"], model_config[model_name]['max_length'])
    else:  # Default to sequence classification
        model_name = "llama"
        tokenizer, model = load_llama_model(model_config[model_name], num_labels, task_type='sequence_classification')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        compute_metrics = compute_metrics_classification
        preprocess_func = lambda examples: preprocess_text(examples, tokenizer, model_config[model_name]['max_length'])
    
    # Load and preprocess data
    dataset_dict = load_data(task_config["name"])
    tokenized_ds = dataset_dict.map(preprocess_func, batched=True)

    # Set up and run trainer
    training_args = TrainingArguments(**training_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_ds["test"])
    trainer.log_metrics("eval", metrics)
    print("Test metrics:", metrics)