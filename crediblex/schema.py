from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class ArticleRecord(BaseModel):
    # The Text
    text: str
    
    # The 4 Targets (Labels) we want to predict
    # 0: Far-Left, 1: Slightly-Left, 2: Center, 3: Slightly-Right, 4: Far-Right
    bias_label: int
    fact_score: float # 0.0 (Fake) to 1.0 (True)
    intent_label: int # 0: News, 1: Opinion, 2: Satire
    # Bug 11: emotion_label is now a JSON multi-hot string "[0,1,0,...]" (28 values)
    emotion_label: str

    @field_validator('fact_score')
    @classmethod
    def check_fact_range(cls, v):
        if v == -1.0: return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("Fact score must be between 0 and 1 (or -1.0 sentinel)")
        return v

    @field_validator('bias_label')
    @classmethod
    def check_bias_range(cls, v):
        if v == -1: return v
        if not (0 <= v <= 4):
            raise ValueError("Bias label must be between 0 and 4 (or -1 sentinel)")
        return v

    @field_validator('intent_label')
    @classmethod
    def check_intent_range(cls, v):
        if v == -1: return v
        if not (0 <= v <= 2):
            raise ValueError("Intent label must be between 0 and 2 (or -1 sentinel)")
        return v