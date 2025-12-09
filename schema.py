from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class ArticleRecord(BaseModel):
    # The Text
    text: str
    
    # The 4 Targets (Labels) we want to predict
    bias_label: int  # 0: Left, 1: Center, 2: Right
    fact_score: float # 0.0 (Fake) to 1.0 (True)
    intent_label: int # 0: News, 1: Opinion, 2: Satire
    emotion_label: int # 0: Neutral, 1: Anger, 2: Joy, etc. (Simplified)

    @field_validator('fact_score')
    @classmethod
    def check_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Fact score must be between 0 and 1")
        return v