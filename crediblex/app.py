# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModel
from config import Config
from model import NewsTrustModel
from text_preprocessing import TextPreprocessor
import pandas as pd
import json
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from scipy.stats import pearsonr
from typing import List

class ArticleRecord(BaseModel):
    text: str
    url: str = None

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
model     = NewsTrustModel(config=Config)
model.load_state_dict(torch.load(Config.SAVE_PATH, map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()
preprocessor = TextPreprocessor()

BIAS_LABELS    = {0: "Left", 1: "Center", 2: "Right"}
INTENT_LABELS  = {0: "News", 1: "Opinion", 2: "Satire"}
EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

@app.get("/health")
def health_check():
    return {"status": "ok", "model": Config.BASE_MODEL_NAME, "device": Config.DEVICE}

@app.post("/predict")
def predict(article: ArticleRecord):
    if article.url:
        try:
            from test2 import extract_article
            text = extract_article(article.url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape URL. Error: {str(e)}")
    else:
        text = article.text

    if not preprocessor.preprocess(text) or len(preprocessor.preprocess(text)) < 10:
        raise HTTPException(status_code=400, detail="Text is too short after preprocessing.")

    inputs = tokenizer(text, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    bias_confidence = max(torch.softmax(outputs['bias'], dim=-1).tolist()[0])
    intent_confidence = max(torch.softmax(outputs['intent'], dim=-1).tolist()[0])

    predictions = {
        "bias": BIAS_LABELS[torch.argmax(outputs['bias']).item()],
        "bias_confidence": bias_confidence,
        "factuality": float(outputs['factuality'].squeeze().item()),
        "intent": INTENT_LABELS[torch.argmax(outputs['intent']).item()],
        "intent_confidence": intent_confidence,
        "emotions": [emotion for emotion, score in zip(EMOTION_LABELS, torch.softmax(outputs['emotion'], dim=-1).tolist()[0]) if score > 0.4],
        "tokens_processed": len(text)
    }

    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)