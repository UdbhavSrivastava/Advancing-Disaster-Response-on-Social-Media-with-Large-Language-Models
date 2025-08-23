import torch
import torch.nn as nn
from transformers import AutoTokenizer, ResNetModel, ViTModel, LlamaConfig, LlamaForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from .modeling_llama import UnmaskingLlamaForSequenceClassification, UnmaskingLlamaForTokenClassification

def load_llama_model(model_config, num_labels, task_type='sequence_classification'):
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_id'])
    tokenizer.pad_token = tokenizer.eos_token

    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}

    if task_type == 'sequence_classification':
        model_class = LlamaForSequenceClassification
        peft_task_type = TaskType.SEQ_CLS
    elif task_type == 'unmasking_sequence_classification':
        model_class = UnmaskingLlamaForSequenceClassification
        peft_task_type = TaskType.SEQ_CLS
    elif task_type == 'token_classification':
        model_class = UnmaskingLlamaForTokenClassification
        peft_task_type = TaskType.TOKEN_CLS
    else:
        raise ValueError(f"Unsupported Llama task type: {task_type}")

    model = model_class.from_pretrained(
        model_config['model_id'],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).bfloat16()

    peft_config_dict = model_config['peft_config']
    peft_config_dict['task_type'] = peft_task_type
    peft_config = LoraConfig(**peft_config_dict)
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model

def load_image_model(model_name):
    if model_name == 'resnet':
        return ResNetModel.from_pretrained('microsoft/resnet-101')
    elif model_name == 'vit':
        return ViTModel.from_pretrained('google/vit-base-patch16-224')
    else:
        raise ValueError(f"Unsupported image model: {model_name}")