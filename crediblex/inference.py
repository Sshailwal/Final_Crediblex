import torch
from transformers import AutoTokenizer
from config import (DEVICE, MODEL_NAME, MAX_LEN, STRIDE, MAX_WINDOWS,
                    SAVE_PATH, LOG_PATH, NUM_BIAS_LABELS,
                    NUM_INTENT_LABELS, NUM_EMOTION_LABELS)
from model import CredibleXModel

_MODEL_CACHE = {"model": None, "tokenizer": None}

def _load_model_once():
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]

    model = CredibleXModel()
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _MODEL_CACHE["model"]     = model
    _MODEL_CACHE["tokenizer"] = tokenizer
    return model, tokenizer

def create_sliding_windows(text, tokenizer):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

    windows = []
    step    = MAX_LEN - 2 - STRIDE   # account for [CLS] and [SEP]
    start   = 0
    while start < len(tokens):
        chunk = tokens[start : start + MAX_LEN - 2]
        ids   = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        pad_len = MAX_LEN - len(ids)
        ids   += [tokenizer.pad_token_id] * pad_len
        mask  = [1] * (len(ids) - pad_len) + [0] * pad_len
        windows.append({
            "input_ids":      ids,
            "attention_mask": mask
        })
        start += step
        if len(windows) >= MAX_WINDOWS:   # HARD CAP — OOM guard
            break

    return windows   # list of dicts, len <= MAX_WINDOWS

def _factuality_label(score: float) -> str:
    if score >= 0.8: return "true"
    if score >= 0.6: return "mostly_true"
    if score >= 0.4: return "half_true"
    if score >= 0.2: return "barely_true"
    return "false"

def run_inference(text: str) -> dict:
    model, tokenizer = _load_model_once()
    amp_enabled      = DEVICE.type == "cuda"   # correct comparison

    windows = create_sliding_windows(text, tokenizer)

    # Collect per-window CLS embeddings and raw head logits
    bias_logits   = []
    fact_scores   = []
    intent_logits = []
    emotion_logits= []

    with torch.no_grad():
        for window in windows:
            input_ids      = torch.tensor([window["input_ids"]],
                                          dtype=torch.long).to(DEVICE)
            attention_mask = torch.tensor([window["attention_mask"]],
                                          dtype=torch.long).to(DEVICE)
            token_type_ids = torch.zeros_like(input_ids)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(input_ids, attention_mask, token_type_ids)

            # Detach and collect — delete input tensors immediately after
            bias_logits.append(outputs["bias"].squeeze(0).cpu())
            fact_scores.append(outputs["factuality"].squeeze(0).cpu())
            intent_logits.append(outputs["intent"].squeeze(0).cpu())
            emotion_logits.append(outputs["emotion"].squeeze(0).cpu())

            del input_ids, attention_mask, token_type_ids, outputs
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # ── HEAD-SPECIFIC POOLING ─────────────────────────────────────
    bias_stack    = torch.stack(bias_logits)     # (W, 5)
    fact_stack    = torch.stack(fact_scores)     # (W, 1)
    intent_stack  = torch.stack(intent_logits)   # (W, 3)
    emotion_stack = torch.stack(emotion_logits)  # (W, 28)

    # Bias + emotion: mean-pool (whole-article signal)
    bias_pooled    = bias_stack.mean(dim=0)      # (5,)
    emotion_pooled = emotion_stack.mean(dim=0)   # (28,)

    # Factuality: mean-pool (holistic article factuality)
    fact_pooled    = fact_stack.mean(dim=0)      # (1,)
    # Intent: max-pool (worst-case window / strongest signal)
    intent_pooled  = intent_stack.max(dim=0).values # (3,)

    # ── DECODE OUTPUTS ────────────────────────────────────────────

    BIAS_LABELS   = ["slightly_left", "left", "center", "right", "slightly_right"]
    INTENT_LABELS = ["news", "opinion", "satire"]

    bias_probs    = torch.softmax(bias_pooled, dim=0).tolist()
    bias_class    = int(torch.argmax(bias_pooled).item())
    bias_conf     = float(bias_probs[bias_class])

    fact_score    = float(fact_pooled.item())

    # ── FACTUALITY CALIBRATION ────────────────────────────────────────────────
    # The model was trained on short LIAR-PLUS statements but receives long
    # articles at inference. This causes regression-to-mean: real scores cluster
    # tightly in [0.20, 0.45]. We apply min-max rescaling to spread the range.
    # Observed floor=0.18 (pure conspiracy), ceiling=0.48 (clean factual text)
    FACT_FLOOR   = 0.18
    FACT_CEILING = 0.48
    fact_score_calibrated = (fact_score - FACT_FLOOR) / (FACT_CEILING - FACT_FLOOR)
    fact_score_calibrated = float(max(0.0, min(1.0, fact_score_calibrated)))

    intent_probs  = torch.softmax(intent_pooled, dim=0).tolist()
    intent_class  = int(torch.argmax(intent_pooled).item())
    intent_conf   = float(intent_probs[intent_class])

    emotion_probs = torch.sigmoid(emotion_pooled).tolist()
    # Return top-3 emotions with scores above 0.3 threshold
    GOEMOTION_LABELS = [
        "admiration","amusement","anger","annoyance","approval","caring",
        "confusion","curiosity","desire","disappointment","disapproval",
        "disgust","embarrassment","excitement","fear","gratitude","grief",
        "joy","love","nervousness","optimism","pride","realization",
        "relief","remorse","sadness","surprise","neutral"
    ]
    top_emotions = sorted(
        [{"label": GOEMOTION_LABELS[i], "score": round(emotion_probs[i], 3)}
         for i in range(28) if emotion_probs[i] > 0.3],
        key=lambda x: x["score"], reverse=True
    )[:3]
    
    # ── CREDIBILITY LOGIC ─────────────────────────────────────────
    bias_data = {"label": BIAS_LABELS[bias_class], "confidence": bias_conf}
    intent_data = {"label": INTENT_LABELS[intent_class], "confidence": intent_conf}
    
    trust_score = _calculate_credibility(fact_score_calibrated, bias_data, intent_data, top_emotions)
    
    if trust_score >= 80: verdict = "Highly Credible"
    elif trust_score >= 60: verdict = "Mostly Credible"
    elif trust_score >= 40: verdict = "Mixed Reliability"
    elif trust_score >= 20: verdict = "Low Credibility"
    else: verdict = "Fabricated / Fake"

    return {
        "windows_processed": len(windows),
        "trust_score": trust_score,
        "verdict": verdict,
        "bias": {
            "label":       bias_data["label"],
            "confidence":  round(bias_data["confidence"], 3),
            "distribution": {BIAS_LABELS[i]: round(bias_probs[i], 3)
                             for i in range(5)}
        },
        "factuality": {
            "score":  round(fact_score_calibrated, 3),
            "label":  _factuality_label(fact_score_calibrated)
        },
        "intent": {
            "label":      intent_data["label"],
            "confidence": round(intent_data["confidence"], 3)
        },
        "emotion": {
            "top": top_emotions
        }
    }

def _calculate_credibility(fact_score, bias_data, intent_data, top_emotions):
    # Base Score from calibrated factuality (0.0 - 1.0 rescaled)
    score = fact_score * 100.0

    # Bias modifier (reward center, penalize extremes)
    if bias_data["label"] == "center":
        score += 10
    elif bias_data["label"] in ["left", "right"]:
        score -= 5
    elif bias_data["label"] in ["slightly_left", "slightly_right"]:
        score -= 2

    # Intent modifier (reward news, penalize satire/opinion)
    if intent_data["label"] == "news":
        score += 10
    elif intent_data["label"] == "opinion":
        score -= 10
    elif intent_data["label"] == "satire":
        score -= 30

    # Emotion modifier (penalize heavy negative emotional framing)
    negative_emotions = ["anger", "annoyance", "disappointment", "disapproval",
                         "disgust", "embarrassment", "fear", "grief", "nervousness",
                         "remorse", "sadness"]
    negative_penalty = sum([8 for e in top_emotions if e["label"] in negative_emotions])
    score -= negative_penalty

    # Bound before applying hard penalty
    score = max(0.0, min(100.0, score))

    # Polite Fake News Penalty:
    # Only trigger if calibrated factuality is below 0.35
    # (previously 0.5 caused ALL articles to be halved since raw scores
    # never exceeded 0.45 due to regression-to-mean)
    if fact_score < 0.35:
        score *= 0.6

    return int(max(0, min(100, score)))

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    short = "This is a short test article."
    long  = "word " * 2000
    w1 = create_sliding_windows(short, tok)
    w2 = create_sliding_windows(long,  tok)
    assert len(w1) >= 1,           "Short text produced no windows"
    assert len(w2) <= MAX_WINDOWS, f"Long text exceeded MAX_WINDOWS cap: {len(w2)}"
    assert len(w1[0]["input_ids"]) == MAX_LEN, "Window length != MAX_LEN"
    print(f"  Short text -> {len(w1)} window(s)")
    print(f"  Long text  -> {len(w2)} window(s) (cap={MAX_WINDOWS})")
    print("Part D inference.py validation passed")