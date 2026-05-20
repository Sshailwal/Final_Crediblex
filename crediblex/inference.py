import pickle
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer
from config import (DEVICE, MODEL_NAME, MAX_LEN, STRIDE, MAX_WINDOWS,
                    SAVE_PATH, LOG_PATH, NUM_BIAS_LABELS,
                    NUM_INTENT_LABELS, NUM_EMOTION_LABELS)
from model import CredibleXModel

BASE_DIR = Path(__file__).resolve().parent
CLASSICAL_MODEL_DIRS = [
    BASE_DIR / "models" / "bias_classifier_v2",
    BASE_DIR / "model",
    BASE_DIR / "models",
    BASE_DIR,
]

_MODEL_CACHE = {"kind": None, "model": None, "tokenizer": None, "labels": None}

def _load_model_once():
    if _MODEL_CACHE["kind"] is not None:
        return _MODEL_CACHE

    save_path = Path(SAVE_PATH)
    if not save_path.is_absolute():
        save_path = BASE_DIR / save_path

    if save_path.exists():
        model = CredibleXModel()
        checkpoint = torch.load(save_path, map_location=DEVICE)
        state = checkpoint.get("model") or checkpoint.get("model_state")
        if state is None:
            raise KeyError(f"Checkpoint {save_path} does not contain model weights")
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        _MODEL_CACHE["kind"] = "neural"
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["tokenizer"] = tokenizer
        return _MODEL_CACHE

    classical_dir = _find_classical_model_dir()
    if classical_dir is not None:
        model_path = classical_dir / "best_model.pkl"
        vectorizer_path = classical_dir / "tfidf_vectorizer.pkl"
        labels_path = classical_dir / "label_encoder.pkl"

        model = None
        vectorizer = None
        labels = {}
        if model_path.exists():
            try:
                with open(model_path, "rb") as fh:
                    model = pickle.load(fh)
            except ModuleNotFoundError as exc:
                print(
                    f"CredibleX warning: could not load {model_path.name} "
                    f"because dependency {exc.name!r} is missing; using heuristic bias fallback."
                )
        if vectorizer_path.exists():
            with open(vectorizer_path, "rb") as fh:
                vectorizer = pickle.load(fh)
        if labels_path.exists():
            with open(labels_path, "rb") as fh:
                labels = pickle.load(fh)

        _MODEL_CACHE["kind"] = "classical"
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["tokenizer"] = vectorizer
        _MODEL_CACHE["labels"] = labels
        print(
            "CredibleX warning: model_v1.pth was not found; "
            f"using {classical_dir.relative_to(BASE_DIR)} fallback."
        )
        return _MODEL_CACHE

    raise FileNotFoundError(
        "No usable model found. Expected model_v1.pth or "
        "best_model.pkl in model/, models/, or models/bias_classifier_v2/."
    )


def _find_classical_model_dir() -> Path | None:
    for candidate in CLASSICAL_MODEL_DIRS:
        if (candidate / "best_model.pkl").exists() or (
            (candidate / "tfidf_vectorizer.pkl").exists()
            and (candidate / "label_encoder.pkl").exists()
        ):
            return candidate
    return None

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
    loaded = _load_model_once()
    if loaded["kind"] == "classical":
        return _run_classical_inference(text, loaded)

    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
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


def _run_classical_inference(text: str, loaded: dict) -> dict:
    """Fallback inference for the exported TF-IDF bias classifier."""
    model = loaded["model"]
    vectorizer = loaded["tokenizer"]

    X = vectorizer.transform([text]) if vectorizer is not None else [text]
    if model is not None:
        pred_id = int(model.predict(X)[0])
        raw_probs = _class_probabilities(model, X, pred_id)
    else:
        pred_id, raw_probs = _heuristic_bias(text)

    bias_labels = ["left", "slightly_left", "center", "slightly_right", "right"]
    bias_label = bias_labels[pred_id] if 0 <= pred_id < len(bias_labels) else "center"
    bias_conf = float(raw_probs[pred_id]) if pred_id < len(raw_probs) else 0.5
    distribution = {
        bias_labels[i]: round(float(raw_probs[i]), 3)
        for i in range(min(len(raw_probs), len(bias_labels)))
    }

    fact_score = _heuristic_factuality(text)
    intent_label, intent_conf = _heuristic_intent(text)
    top_emotions = _heuristic_emotions(text)
    trust_score = _calculate_credibility(
        fact_score,
        {"label": bias_label, "confidence": bias_conf},
        {"label": intent_label, "confidence": intent_conf},
        top_emotions,
    )

    if trust_score >= 80:
        verdict = "Highly Credible"
    elif trust_score >= 60:
        verdict = "Mostly Credible"
    elif trust_score >= 40:
        verdict = "Mixed Reliability"
    elif trust_score >= 20:
        verdict = "Low Credibility"
    else:
        verdict = "Fabricated / Fake"

    return {
        "windows_processed": 1,
        "trust_score": trust_score,
        "verdict": verdict,
        "bias": {
            "label": bias_label,
            "confidence": round(bias_conf, 3),
            "distribution": distribution,
        },
        "factuality": {
            "score": round(fact_score, 3),
            "label": _factuality_label(fact_score),
        },
        "intent": {
            "label": intent_label,
            "confidence": round(intent_conf, 3),
        },
        "emotion": {
            "top": top_emotions,
        },
    }


def _class_probabilities(model, X, pred_id: int) -> list[float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return [float(p) for p in probs]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        scores = [float(s) for s in scores]
        max_score = max(scores)
        exps = [pow(2.718281828, s - max_score) for s in scores]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    probs = [0.0] * 5
    probs[pred_id] = 1.0
    return probs


def _heuristic_factuality(text: str) -> float:
    lower = text.lower()
    score = 0.62

    sensational = [
        "shocking", "miracle", "you won't believe", "exposed", "secret",
        "conspiracy", "hoax", "must share", "forward this", "breaking!!!",
    ]
    evidence = [
        "according to", "reported", "data", "study", "court", "official",
        "statement", "ministry", "police", "agency", "commission",
    ]

    score += min(0.18, 0.03 * sum(1 for word in evidence if word in lower))
    score -= min(0.28, 0.05 * sum(1 for word in sensational if word in lower))
    if re.search(r"https?://|www\.", lower):
        score += 0.03
    if len(text.split()) < 80:
        score -= 0.08
    if sum(1 for ch in text if ch == "!") >= 3:
        score -= 0.08

    return float(max(0.05, min(0.95, score)))


def _heuristic_intent(text: str) -> tuple[str, float]:
    lower = text.lower()
    satire_terms = ["satire", "parody", "spoof", "humor", "comedy"]
    opinion_terms = [
        "opinion", "editorial", "i think", "we believe", "should",
        "must", "ought", "column", "viewpoint",
    ]

    if any(term in lower for term in satire_terms):
        return "satire", 0.82
    if any(term in lower for term in opinion_terms):
        return "opinion", 0.72
    return "news", 0.68


def _heuristic_bias(text: str) -> tuple[int, list[float]]:
    lower = text.lower()
    left_terms = [
        "worker", "workers", "union", "progressive", "inequality",
        "public healthcare", "welfare", "climate justice", "billionaire",
    ]
    right_terms = [
        "lower taxes", "border", "national identity", "free market",
        "traditional", "conservative", "law and order", "small government",
    ]
    far_left_terms = ["seize power", "capitalist", "revolution", "far left"]
    far_right_terms = ["radical left", "open borders", "globalist", "far right"]

    score = 0
    score -= 2 * sum(1 for term in far_left_terms if term in lower)
    score -= sum(1 for term in left_terms if term in lower)
    score += sum(1 for term in right_terms if term in lower)
    score += 2 * sum(1 for term in far_right_terms if term in lower)

    if score <= -3:
        pred_id = 0
    elif score <= -1:
        pred_id = 1
    elif score >= 3:
        pred_id = 4
    elif score >= 1:
        pred_id = 3
    else:
        pred_id = 2

    probs = [0.08, 0.14, 0.56, 0.14, 0.08]
    probs[pred_id] = 0.68
    remainder = 0.32
    for idx in range(5):
        if idx != pred_id:
            probs[idx] = remainder / 4
    return pred_id, probs


def _heuristic_emotions(text: str) -> list[dict]:
    lower = text.lower()
    buckets = {
        "anger": ["outrage", "angry", "furious", "corrupt", "betrayal"],
        "fear": ["fear", "threat", "danger", "panic", "risk"],
        "sadness": ["tragic", "grief", "sad", "death", "loss"],
        "optimism": ["hope", "progress", "improve", "success", "win"],
        "neutral": [],
    }

    found = []
    for label, words in buckets.items():
        if label == "neutral":
            continue
        hits = sum(1 for word in words if word in lower)
        if hits:
            found.append({"label": label, "score": round(min(0.9, 0.35 + hits * 0.12), 3)})

    if not found:
        found.append({"label": "neutral", "score": 0.5})

    return sorted(found, key=lambda item: item["score"], reverse=True)[:3]

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
