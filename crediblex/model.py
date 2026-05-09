import torch
import torch.nn as nn
from transformers import AutoModel
import config

class NewsTrustModel(nn.Module):
    """
    Multi-task DeBERTa-v3-base model for news credibility scoring.
    Optimized for Safe Mode (6GB VRAM).
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(model_name)
        hidden           = self.backbone.config.hidden_size

        # Shared regularisation
        self.dropout = nn.Dropout(dropout)

        # Classification / regression heads
        self.bias_head    = nn.Linear(hidden, config.N_BIAS_CLASSES)
        self.fact_head    = nn.Linear(hidden, 1)
        self.intent_head  = nn.Linear(hidden, 2)
        self.emotion_head = nn.Linear(hidden, 7)   # 7 Ekman classes (mapped from GoEmotions 28)

    def configure_for_training(self):
        """
        Apply memory optimizations: gradient checkpointing and layer freezing.
        """
        # 1. Enable gradient checkpointing
        if config.GRADIENT_CHECKPOINTING:
            self.backbone.gradient_checkpointing_enable()
            print("[INFO] Gradient checkpointing ENABLED")

        # 2. Freeze bottom layers
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < config.FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print("[INFO] Froze embeddings and bottom {} layers".format(config.FREEZE_LAYERS))

        # 3. Print parameter summary
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Trainable: {:,} / {:,} params ({:.1f}%)".format(
            trainable_params, total_params, trainable_params/total_params*100))

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor,
                   attention_mask:   torch.Tensor) -> torch.Tensor:
        """
        Average non-padding token embeddings weighted by the attention mask.
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        sum_emb  = torch.sum(token_embeddings * mask_expanded, dim=1) # (B, H)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)    # (B, 1)
        return sum_emb / sum_mask                                      # (B, H)

    @staticmethod
    def create_sliding_windows(text: str, tokenizer, max_len: int):
        """
        Divide long text into overlapping windows for processing.
        Returns (list of input_ids tensors, list of attention_mask tensors).
        """
        # 1. Tokenize entire text without truncation
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
        
        if not token_ids:
            # Handle empty/tiny text
            enc = tokenizer("", max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
            return [enc["input_ids"][0]], [enc["attention_mask"][0]]

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        
        # Max capacity for text chunk is max_len - 2 (CLS + SEP)
        inner_len = max_len - 2
        stride    = getattr(config, "SLIDING_STRIDE", inner_len // 2)
        
        input_ids_list = []
        attention_mask_list = []
        
        start = 0
        while start < len(token_ids):
            end   = min(start + inner_len, len(token_ids))
            chunk = token_ids[start:end]
            
            # [CLS] + chunk + [SEP] + [PAD]...
            ids = [cls_id] + chunk + [sep_id]
            pad_len = max_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids += [pad_id] * pad_len
            
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(mask, dtype=torch.long))
            
            if end == len(token_ids):
                break
            start += stride
            
        return input_ids_list, attention_mask_list

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        """
        Standard forward pass for a single window/batch.
        """
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)

        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        return {
            "bias":       self.bias_head(pooled),
            "factuality": torch.sigmoid(self.fact_head(pooled)),
            "intent":     self.intent_head(pooled),
            "emotion":    self.emotion_head(pooled),
        }