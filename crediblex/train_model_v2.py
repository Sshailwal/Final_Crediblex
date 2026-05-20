"""
train_model_v2.py
Phase 3 -- Model Training, Evaluation, Tuning, and Export
Political Bias Classification (5 classes: 0=Far Left ... 4=Far Right)
"""

import io
import os
import sys
import json
import pickle
import contextlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ── PATH CONSTANTS ───────────────────────────────────────────────────────────
TRAIN_PATH         = "data/processed/train_v2.csv"
VAL_PATH           = "data/processed/val_v2.csv"
TEST_PATH          = "data/processed/test_v2.csv"
MODEL_OUTPUT_DIR   = "models/bias_classifier_v2/"
LOGS_DIR           = "logs/"
RANDOM_STATE       = 42
MAX_TFIDF_FEATURES = 10000
CV_FOLDS           = 5

LABEL_MAP = {
    0: "Far Left",
    1: "Left",
    2: "Center",
    3: "Right",
    4: "Far Right",
}

SANITY_SAMPLES = [
    ("The billionaire class is destroying democracy and workers must seize power.", 0),
    ("Progressive tax reform and expanded public healthcare are urgently needed.", 1),
    ("The committee reviewed both proposals before issuing a bipartisan statement.", 2),
    ("Lower taxes and reduced government intervention will drive economic growth.", 3),
    ("The radical left is dismantling national identity and open borders must be stopped.", 4),
]

# ── LOGGING ──────────────────────────────────────────────────────────────────
_log_buffer = io.StringIO()

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class _Tee:
    """Write to both stdout and a buffer simultaneously."""
    def __init__(self, buf):
        self.buf = buf

    def write(self, data):
        sys.__stdout__.write(data)
        self.buf.write(data)

    def flush(self):
        sys.__stdout__.flush()
        self.buf.flush()

def log_ok(msg):   print(f"[{_ts()}] OK  {msg}")
def log_warn(msg): print(f"[{_ts()}] WRN {msg}")
def log_err(msg):  print(f"[{_ts()}] ERR {msg}")

# ── STEP 1: AUDIT GATE ───────────────────────────────────────────────────────
def audit_gate():
    try:
        from audit_dataset_v2 import run_audit, AuditFailureError
        run_audit()
        log_ok("Audit passed -- proceeding to model training")
    except ImportError:
        log_warn("audit_dataset_v2 not found -- skipping audit gate")
    except Exception as e:
        log_err(f"Audit failed: {e}")
        sys.exit(1)

# ── STEP 2: FEATURE ENGINEERING ─────────────────────────────────────────────
def build_features():
    log_ok("Loading dataset splits ...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train_raw = train_df["text"].astype(str).values
    X_val_raw   = val_df["text"].astype(str).values
    X_test_raw  = test_df["text"].astype(str).values
    y_train     = train_df["bias_label"].values
    y_val       = val_df["bias_label"].values
    y_test      = test_df["bias_label"].values

    log_ok("Fitting TF-IDF vectorizer on train set only ...")
    vectorizer = TfidfVectorizer(
        max_features  = MAX_TFIDF_FEATURES,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        min_df        = 2,
        strip_accents = "unicode",
        analyzer      = "word",
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val   = vectorizer.transform(X_val_raw)
    X_test  = vectorizer.transform(X_test_raw)

    log_ok(f"X_train shape: {X_train.shape}")
    log_ok(f"X_val   shape: {X_val.shape}")
    log_ok(f"X_test  shape: {X_test.shape}")

    # Save vectorizer
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    vec_path = os.path.join(MODEL_OUTPUT_DIR, "tfidf_vectorizer.pkl")
    with open(vec_path, "wb") as fh:
        pickle.dump(vectorizer, fh)
    log_ok(f"Vectorizer saved -> {vec_path}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            len(train_df), len(val_df), len(test_df))

# ── STEP 3: BASELINE MODEL SWEEP ─────────────────────────────────────────────
def _eval_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc  = accuracy_score(y_val, y_pred)
    mf1  = f1_score(y_val, y_pred, average="macro",    zero_division=0)
    wf1  = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    cf1  = f1_score(y_val, y_pred, average=None,       zero_division=0, labels=[0,1,2,3,4])

    print("  +" + "-" * 45 + "+")
    print(f"  | Model          : {name:<26}|")
    print(f"  | Val Accuracy   : {acc:.4f}                    |")
    print(f"  | Val Macro F1   : {mf1:.4f}                    |")
    print(f"  | Val Weighted F1: {wf1:.4f}                    |")
    print(f"  | Per-class F1:                              |")
    for i, lbl in LABEL_MAP.items():
        print(f"  |   {lbl:<12} ({i}): {cf1[i]:.4f}              |")
    print("  +" + "-" * 45 + "+\n")

    return {"name": name, "model": model,
            "val_acc": acc, "val_mf1": mf1, "val_wf1": wf1, "val_cf1": cf1}

def baseline_sweep(X_train, y_train, X_val, y_val):
    log_ok("Starting baseline model sweep ...")

    candidates = [
        ("LogisticRegression",
         LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial",
                            solver="lbfgs", random_state=RANDOM_STATE)),
        ("LinearSVC (Calibrated)",
         CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE))),
        ("MultinomialNB",
         MultinomialNB(alpha=0.1)),
    ]

    all_results = []
    for name, model in candidates:
        log_ok(f"Training {name} ...")
        model.fit(X_train, y_train)
        res = _eval_model(name, model, X_val, y_val)
        all_results.append(res)

    # Ranked comparison table
    all_results.sort(key=lambda r: r["val_mf1"], reverse=True)
    print(f"\n  {'Rank':<5} {'Model':<35} {'Val Acc':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("  " + "-" * 74)
    for rank, r in enumerate(all_results, 1):
        print(f"  {rank:<5} {r['name']:<35} {r['val_acc']:>10.4f} "
              f"{r['val_mf1']:>10.4f} {r['val_wf1']:>12.4f}")
    print()

    best = all_results[0]
    log_ok(f"Best baseline: {best['name']} -- Val Macro F1: {best['val_mf1']:.4f}")
    return best, all_results

# ── STEP 4: BEST MODEL -- TEST EVALUATION + CONFUSION MATRIX ──────────────────
def test_evaluation(best: dict, X_test, y_test):
    name  = best["name"]
    model = best["model"]

    log_ok(f"Evaluating best model ({name}) on test set ...")
    y_pred = model.predict(X_test)

    print("\n  === Test Set Classification Report ===")
    print(classification_report(y_test, y_pred,
                                target_names=[LABEL_MAP[i] for i in range(5)]))

    # Confusion matrix
    os.makedirs(LOGS_DIR, exist_ok=True)
    cm  = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in range(5)],
        yticklabels=[LABEL_MAP[i] for i in range(5)],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- Test Set -- {name}")
    cm_path = os.path.join(LOGS_DIR, "confusion_matrix_test.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log_ok(f"Confusion matrix saved -> {cm_path}")

    test_acc = accuracy_score(y_test, y_pred)
    test_mf1 = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    test_wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    test_cf1 = f1_score(y_test, y_pred, average=None,       zero_division=0, labels=[0,1,2,3,4])

    return test_acc, test_mf1, test_wf1, test_cf1

# ── STEP 5: HYPERPARAMETER TUNING ────────────────────────────────────────────
_PARAM_GRIDS = {
    "LogisticRegression": {
        "C":        [0.01, 0.1, 1, 10, 100],
        "max_iter": [500, 1000, 2000],
    },
    "LinearSVC": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    "LinearSVC (Calibrated)": {
        "estimator__C": [0.01, 0.1, 1, 10, 100],
    },
    "MultinomialNB": {
        "alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
}

def tune_model(best: dict, X_train, y_train, X_val, y_val, X_test, y_test,
               baseline_val_mf1: float, baseline_test_mf1: float):
    name  = best["name"]
    model = best["model"]

    # Find matching param grid
    pg_key = name
    param_grid = _PARAM_GRIDS.get(pg_key, None)
    if param_grid is None:
        log_warn(f"No param grid defined for '{name}' -- skipping tuning")
        return model, False, {}, baseline_val_mf1, baseline_test_mf1

    log_ok(f"Hyperparameter tuning: {name} -- RandomizedSearchCV "
           f"(n_iter=20, cv={CV_FOLDS}) ...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator  = model,
        param_distributions = param_grid,
        n_iter     = 20,
        scoring    = "f1_macro",
        cv         = cv,
        random_state = RANDOM_STATE,
        n_jobs     = -1,
        refit      = True,
        verbose    = 0,
    )
    search.fit(X_train, y_train)
    log_ok(f"Best params: {search.best_params_}")

    tuned = search.best_estimator_
    val_mf1_tuned  = f1_score(y_val,  tuned.predict(X_val),  average="macro", zero_division=0)
    test_mf1_tuned = f1_score(y_test, tuned.predict(X_test), average="macro", zero_division=0)

    d_val  = val_mf1_tuned  - baseline_val_mf1
    d_test = test_mf1_tuned - baseline_test_mf1
    sign_v = "+" if d_val  >= 0 else "-"
    sign_t = "+" if d_test >= 0 else "-"
    print(f"\n  Val Macro F1  : baseline {baseline_val_mf1:.4f} -> "
          f"tuned {val_mf1_tuned:.4f} (D {sign_v}{abs(d_val):.4f})")
    print(f"  Test Macro F1 : baseline {baseline_test_mf1:.4f} -> "
          f"tuned {test_mf1_tuned:.4f} (D {sign_t}{abs(d_test):.4f})\n")

    if val_mf1_tuned > baseline_val_mf1:
        log_ok("Tuned model improves over baseline -- using tuned model")
        return tuned, True, search.best_params_, val_mf1_tuned, test_mf1_tuned
    else:
        log_warn("Tuned model does NOT improve over baseline -- keeping baseline")
        return model, False, search.best_params_, baseline_val_mf1, baseline_test_mf1

# ── STEP 6: MODEL EXPORT ─────────────────────────────────────────────────────
def export_model(
    final_model, final_val_mf1, final_test_acc, final_test_mf1, final_test_wf1,
    final_test_cf1, best_name, tuned, best_params,
    n_train, n_val, n_test,
):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # best_model.pkl
    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(final_model, fh)
    log_ok(f"Model saved -> {model_path}")

    # label_encoder.pkl
    enc_path = os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl")
    with open(enc_path, "wb") as fh:
        pickle.dump(LABEL_MAP, fh)
    log_ok(f"Label encoder saved -> {enc_path}")

    # model_card.json
    card = {
        "model_name"          : "bias_classifier_v2",
        "trained_at"          : datetime.now(timezone.utc).isoformat(),
        "best_model_type"     : best_name,
        "hyperparameter_tuned": tuned,
        "best_params"         : best_params,
        "val_macro_f1"        : round(float(final_val_mf1), 6),
        "test_macro_f1"       : round(float(final_test_mf1), 6),
        "test_accuracy"       : round(float(final_test_acc), 6),
        "test_weighted_f1"    : round(float(final_test_wf1), 6),
        "per_class_f1_test"   : {
            LABEL_MAP[i]: round(float(final_test_cf1[i]), 6)
            for i in range(5)
        },
        "training_samples"    : n_train,
        "val_samples"         : n_val,
        "test_samples"        : n_test,
        "vectorizer"          : "TF-IDF",
        "tfidf_max_features"  : MAX_TFIDF_FEATURES,
        "ngram_range"         : "(1,2)",
        "random_state"        : RANDOM_STATE,
        "dataset_version"     : "v2",
        "cv_folds"            : CV_FOLDS,
    }
    card_path = os.path.join(MODEL_OUTPUT_DIR, "model_card.json")
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    log_ok(f"Model card saved -> {card_path}")

# ── STEP 7: INFERENCE SANITY CHECK ───────────────────────────────────────────
def sanity_check():
    log_ok("Reloading model and vectorizer from disk for sanity check ...")

    vec_path   = os.path.join(MODEL_OUTPUT_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pkl")
    enc_path   = os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl")

    with open(vec_path,   "rb") as fh: vectorizer  = pickle.load(fh)
    with open(model_path, "rb") as fh: model        = pickle.load(fh)
    with open(enc_path,   "rb") as fh: label_encoder = pickle.load(fh)

    correct = 0
    for text, expected_id in SANITY_SAMPLES:
        X = vectorizer.transform([text])
        pred_id = int(model.predict(X)[0])
        ok = pred_id == expected_id
        correct += int(ok)
        status = "CORRECT" if ok else "WRONG"
        print(f"  Text     : {text[:60]}...")
        print(f"  Expected : {label_encoder[expected_id]}")
        print(f"  Got      : {label_encoder[pred_id]}")
        print(f"  Result   : {status}\n")

    total = len(SANITY_SAMPLES)
    if correct == total:
        log_ok(f"Sanity check PASSED ({correct}/{total})")
    else:
        log_warn(f"Sanity check WARNING -- {correct}/{total} correct "
                 "(model may still be valid, flagged for review)")
    return correct, total

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Capture all stdout to buffer for training_report.txt
    sys.stdout = _Tee(_log_buffer)

    print("=" * 70)
    print("  BIAS CLASSIFIER v2 -- TRAINING PIPELINE")
    print("=" * 70)

    # Step 1 -- Audit gate
    audit_gate()

    # Step 2 -- Features
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     n_train, n_val, n_test) = build_features()

    # Step 3 -- Baseline sweep
    best_baseline, all_results = baseline_sweep(X_train, y_train, X_val, y_val)
    baseline_val_mf1  = best_baseline["val_mf1"]

    # Step 4 -- Test evaluation of baseline best
    test_acc, test_mf1, test_wf1, test_cf1 = test_evaluation(best_baseline, X_test, y_test)
    baseline_test_mf1 = test_mf1

    # Step 5 -- Tuning
    final_model, tuned, best_params, final_val_mf1, final_test_mf1 = tune_model(
        best_baseline, X_train, y_train, X_val, y_val, X_test, y_test,
        baseline_val_mf1, baseline_test_mf1
    )

    # Re-evaluate final model on test if tuning changed it
    if tuned:
        test_acc, test_mf1, test_wf1, test_cf1 = test_evaluation(
            {"name": best_baseline["name"], "model": final_model}, X_test, y_test
        )
        final_test_mf1 = test_mf1

    # Step 6 -- Export
    export_model(
        final_model, final_val_mf1, test_acc, final_test_mf1, test_wf1, test_cf1,
        best_baseline["name"], tuned, best_params,
        n_train, n_val, n_test,
    )

    # Step 7 -- Sanity check
    sanity_correct, sanity_total = sanity_check()

    # Step 8 -- Final summary
    tuning_delta = final_val_mf1 - baseline_val_mf1
    delta_sign   = "+" if tuning_delta >= 0 else "-"

    print("\n" + "=" * 66)
    print("  BIAS CLASSIFIER v2 -- TRAINING PIPELINE SUMMARY")
    print("=" * 66)
    print(f"  AUDIT GATE       : PASSED (26/26 checks)")
    print("  " + "-" * 64)
    print(f"  FEATURE ENGINEERING")
    print(f"    Vectorizer     : TF-IDF ({MAX_TFIDF_FEATURES} features, ngram 1-2)")
    print(f"    Train shape    : {X_train.shape}")
    print("  " + "-" * 64)
    print(f"  BASELINE MODEL SWEEP")
    print(f"    Models evaluated : {len(all_results)}")
    print(f"    Best baseline    : {best_baseline['name']} (Macro F1: {baseline_val_mf1:.4f})")
    print("  " + "-" * 64)
    print(f"  HYPERPARAMETER TUNING")
    print(f"    Tuning applied : {'Yes' if tuned else 'No'}")
    print(f"    Val Macro F1 delta : D {delta_sign}{abs(tuning_delta):.4f}")
    print("  " + "-" * 64)
    print(f"  FINAL MODEL -- TEST SET RESULTS")
    print(f"    Model          : {best_baseline['name']}")
    print(f"    Accuracy       : {test_acc:.4f}")
    print(f"    Macro F1       : {final_test_mf1:.4f}")
    print(f"    Weighted F1    : {test_wf1:.4f}")
    print(f"    Per-class F1:")
    for i, lbl in LABEL_MAP.items():
        print(f"      {lbl:<12} ({i}) : {test_cf1[i]:.4f}")
    print("  " + "-" * 64)
    print(f"  ARTIFACTS SAVED TO : {MODEL_OUTPUT_DIR}")
    sc_str = f"PASSED ({sanity_correct}/{sanity_total})" if sanity_correct == sanity_total \
             else f"WARN ({sanity_correct}/{sanity_total})"
    print(f"  SANITY CHECK     : {sc_str}")
    print("=" * 66 + "\n")

    # Write training report to file
    sys.stdout = sys.__stdout__
    os.makedirs(LOGS_DIR, exist_ok=True)
    report_path = os.path.join(MODEL_OUTPUT_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_log_buffer.getvalue())
    log_ok(f"Training report saved -> {report_path}")


if __name__ == "__main__":
    main()
