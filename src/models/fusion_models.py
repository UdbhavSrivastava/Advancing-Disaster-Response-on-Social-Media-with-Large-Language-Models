import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetModel, ViTModel

class MultimodalConcatenation(nn.Module):
    def __init__(self, text_model, image_model, fusion_hidden_dim, num_labels):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        text_hidden_dim = text_model.config.hidden_size
        image_hidden_dim = image_model.config.hidden_size if hasattr(image_model.config, "hidden_size") else 2048
        
        fusion_input_size = text_hidden_dim + image_hidden_dim
        self.fusion_proj = nn.Linear(fusion_input_size, fusion_hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(fusion_hidden_dim, num_labels)
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_model.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
        text_embeds = text_outputs.hidden_states[-1].mean(dim=1)
        
        image_outputs = self.image_model(pixel_values=pixel_values)
        if hasattr(image_outputs, 'pooler_output'):
            image_embeds = image_outputs.pooler_output
        else:
            image_embeds = F.adaptive_avg_pool2d(image_outputs.last_hidden_state, (1, 1)).view(image_outputs.last_hidden_state.size(0), -1)

        combined = torch.cat([text_embeds, image_embeds], dim=1)
        fused = self.activation(self.fusion_proj(combined))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}


class CrossModalFusionAttention(nn.Module):
    def __init__(self, text_model, image_model, fusion_dim, num_labels, num_heads=8):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        text_hidden_dim = text_model.config.hidden_size
        image_hidden_dim = image_model.config.hidden_size if hasattr(image_model.config, "hidden_size") else 2048
        
        self.text_proj = nn.Linear(text_hidden_dim, fusion_dim)
        self.image_proj = nn.Linear(image_hidden_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(fusion_dim, num_labels)
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_model.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
        text_hidden = text_outputs.hidden_states[-1].float()
        text_proj = self.text_proj(text_hidden)
        
        image_outputs = self.image_model(pixel_values)
        image_hidden = image_outputs.last_hidden_state.float()

        if image_hidden.dim() == 4:
            B, C, H, W = image_hidden.shape
            image_seq = image_hidden.view(B, C, H * W).transpose(1, 2)
        else:
            image_seq = image_hidden
        
        image_proj = self.image_proj(image_seq)
        
        attn_output, _ = self.cross_attn(query=text_proj, key=image_proj, value=image_proj)
        fused_text = text_proj + attn_output
        fused_representation = fused_text.mean(dim=1)
        logits = self.classifier(fused_representation)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}