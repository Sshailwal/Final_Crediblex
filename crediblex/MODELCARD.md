# Rediblex Model Card

## Overview
- **Model name:** Rediblex V1
- **Architecture:** DeBERTa-based multi-task classifier
- **Tasks:** Bias detection, Factuality scoring, Intent classification, Emotion detection
- **Checkpoint:** rediblex_v1.pth
- **Training date:** 2026-05-08

## Training Data
- **Total rows:** 11,132 (after augmentation, bias rows ~3,000)
- **Bias:** BABE dataset — per-sentence type annotations only
    - Source: `mediabiasgroup/babe` (HuggingFace)
    - Augmentation: EN→FR→EN back-translation (Helsinki-NLP/opus-mt)
    - Final rows: ~1,000 per class after augmentation
- **Factuality:** UKPLab/liar (binary, 0=true→1.0, 1=false→0.0)
    - Capped at 3,000 rows
- **Intent:** GonzaloA/fake_news (Satire vs News, binary)
    - Capped at 3,000 rows
- **Emotion:** GoEmotions → 6 Ekman classes
    - Neutral capped at 600 rows, total ~2,906 rows

## Performance
| Head | Metric | Value | Notes |
| ----------- | ------------------- | ------------- | ----- |
| **Bias** | Held-out accuracy | 69.0% (8/10) | 33% random baseline |
| **Factuality** | MAE / pred range | 0.2530 / 0.26–0.85 | HuberLoss |
| **Intent** | Held-out accuracy | 100% (10/10) | Binary News vs Satire |
| **Emotion** | Active classes | 4/7 @ 0.3 | Joy/Sadness/Anger/Neutral |

## Recommended Operating Thresholds
- **Emotion:** threshold = 0.3 (not 0.5 — model uses `pos_weight=3`)
- **Factuality:** reliable range 0.3–0.8, use caution outside this range
- **Bias:** high confidence on strongly-worded political text, lower confidence on factual/neutral reporting
- **Intent:** no threshold needed, argmax on 2-class softmax output

## Known Limitations
1. **Bias head** — Right texts are occasionally predicted as Left when anti-Trump language is present (conflated vocabulary in partisan fighting).
2. **Bias head** — trained on English political news only, unreliable on non-political or non-English text.
3. **Emotion head** — Fear, Surprise, and Disgust are not activating at threshold=0.3 due to insufficient training samples (<50 per class).
4. **Factuality head** — score compression at extremes (near 0.0 and 1.0) due to regression toward mean. MAE degrades outside the 0.3–0.8 range.
5. **All heads** — trained on news/political text only. Performance on social media, academic, or conversational text is untested and potentially unreliable.

## What Was Intentionally Excluded
- **SemEval data:** excluded from training, reserved for validation only
- **cc_news synthetic padding:** removed in v5 dataset rewrite
- **Outlet-level bias labels:** replaced with per-sentence BABE annotations
- **GoEmotions neutral class:** hard-capped to prevent head collapse

## How to Run Inference

```python
from model import NewsTrustModel # RediblexModel
import torch
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewsTrustModel("microsoft/deberta-v3-base").to(device)
model.load_state_dict(torch.load("rediblex_v1.pth", map_location=device)["model_state_dict"])
model.eval()
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

text = "Your input sentence here."
# Note: In production, use the sliding window encoder and pooler from train.py
# outputs = model(text_embeddings)

# Outputs expected behavior:
# emotion: apply threshold=0.3 to sigmoid probabilities
# bias: argmax → 0=Left, 1=Center, 2=Right
# intent: argmax → 0=News, 1=Satire
# factuality: raw float 0.0–1.0
```
