import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
# This the model file
class NewsTrustModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # HEAD 1: Political Bias (3 Classes: Left, Center, Right)
        self.bias_head = nn.Linear(self.hidden_size, 3)
        
        # HEAD 2: Factuality (1 Score: 0.0 to 1.0)
        self.fact_head = nn.Linear(self.hidden_size, 1)
        
        # HEAD 3: Intent (3 Classes: News, Opinion, Satire)
        self.intent_head = nn.Linear(self.hidden_size, 3)
        
        # HEAD 4: Emotion (28 Classes from GoEmotions)
        self.emotion_head = nn.Linear(self.hidden_size, 28)

    def forward(self, input_ids, attention_mask):
        # 1. Read Text
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Get Summary Vector ([CLS] token)
        # We take the first token of the sequence which represents the whole sentence
        doc_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 3. Make Predictions
        return {
            'bias': self.bias_head(doc_embedding),
            'factuality': torch.sigmoid(self.fact_head(doc_embedding)), # Squash to 0-1
            'intent': self.intent_head(doc_embedding),
            'emotion': self.emotion_head(doc_embedding)
        }
