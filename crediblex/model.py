import torch
import torch.nn as nn
from transformers import AutoModel
import config


class NewsTrustModel(nn.Module):
    """
    Multi-task DeBERTa-v3-base model for news credibility scoring.

    Four output heads trained jointly:
      bias_head      → N_BIAS_CLASSES-class (Far-Left / Slightly Left / Center / Slightly Right / Far-Right)
      fact_head      → regression scalar in [0, 1]  (factuality)
      intent_head    → 3-class (News / Opinion / Satire)
      emotion_head   → 28-class (GoEmotions)

    Architecture notes
    ------------------
    • Mean pooling over the full token sequence, masked to real tokens.
      This is measurably better than [CLS] pooling for DeBERTa-v3 because
      DeBERTa trains [CLS] with disentangled attention, not as a pooling token.
    • Shared dropout(0.1) applied to the pooled embedding before every head
      to regularise all tasks equally and prevent overfitting.
    • Bias head is 5-class (config.N_BIAS_CLASSES) for finer-grained
      Left ↔ Center ↔ Right detection.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(model_name)
        hidden           = self.backbone.config.hidden_size

        # Shared regularisation — applied AFTER mean pooling, BEFORE every head
        self.dropout = nn.Dropout(dropout)

        # ── Classification / regression heads ──────────────────────────────
        # bias_head uses config.N_BIAS_CLASSES (= 5):
        #   0 = Far Left | 1 = Slightly Left | 2 = Center | 3 = Slightly Right | 4 = Far Right
        self.bias_head    = nn.Linear(hidden, config.N_BIAS_CLASSES)
        self.fact_head    = nn.Linear(hidden, 1)   # factuality regression
        self.intent_head  = nn.Linear(hidden, 3)   # News / Opinion / Satire
        self.emotion_head = nn.Linear(hidden, 28)  # GoEmotions 28-class

    # ───────── Attention-masked mean pooling ───────────────────────────────
    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor,
                   attention_mask:   torch.Tensor) -> torch.Tensor:
        """
        Average non-padding token embeddings weighted by the attention mask.

        Parameters
        ----------
        token_embeddings : (B, L, H)
        attention_mask   : (B, L)   — 1 for real tokens, 0 for padding

        Returns
        -------
        pooled : (B, H)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        sum_emb  = torch.sum(token_embeddings * mask_expanded, dim=1) # (B, H)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)    # (B, 1)
        return sum_emb / sum_mask                                      # (B, H)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        # 1. Backbone
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)

        # 2. Mean pool over real tokens (DeBERTa-v3 best practice)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        # 3. Shared dropout
        pooled = self.dropout(pooled)

        # 4. Per-task heads
        return {
            "bias":       self.bias_head(pooled),
            "factuality": torch.sigmoid(self.fact_head(pooled)),  # → [0, 1]
            "intent":     self.intent_head(pooled),
            "emotion":    self.emotion_head(pooled),
        }