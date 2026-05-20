"""
Evaluation script for CredibleX v2 — Indian Political Bias Classifier.

Runs inference on data/test.csv using the best checkpoint, then prints:
    - sklearn classification_report (per-class precision/recall/F1)
    - Macro F1 score
    - Cohen's kappa
    - Saves a seaborn confusion matrix heatmap as confusion_matrix_indian.png

Usage:
    python src/evaluate_indian.py
    python src/evaluate_indian.py --checkpoint checkpoints/model_v2_indian.pth
    python src/evaluate_indian.py --checkpoint checkpoints/model_v2_indian.pth --test-csv data/test.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import CredibleXv2, load_checkpoint
from train_indian import IndianBiasDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_TARGET_NAMES = ["Far Left", "Left", "Center", "Right", "Far Right"]
_DEFAULT_CKPT = "checkpoints/model_v2_indian.pth"
_DEFAULT_CSV  = "data/test.csv"
_DEFAULT_OUT  = "confusion_matrix_indian.png"
_MAX_LEN      = 256
_BATCH_SIZE   = 32


def run_evaluation(
    checkpoint_path: str = _DEFAULT_CKPT,
    test_csv: str = _DEFAULT_CSV,
    output_png: str = _DEFAULT_OUT,
) -> None:
    """
    Run full evaluation on the test split and print/save results.

    Steps:
        1. Load checkpoint and rebuild CredibleXv2.
        2. Tokenise and batch test data with IndianBiasDataset.
        3. Run forward pass (no_grad) to collect predictions.
        4. Print sklearn classification_report, macro F1, Cohen's kappa.
        5. Save confusion matrix heatmap as a PNG.

    Args:
        checkpoint_path: Path to the .pth checkpoint saved by train_indian.py.
        test_csv:        Path to the test split CSV.
        output_png:      Output path for the confusion matrix image.

    Raises:
        FileNotFoundError: If checkpoint or test CSV not found.
    """
    # ── Validate files ───────────────────────────────────────────────────────
    for path, label in [(checkpoint_path, "Checkpoint"), (test_csv, "Test CSV")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on: %s", device)

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})
    backbone = cfg.get("model_name", "xlm-roberta-base")

    model = CredibleXv2(backbone=backbone)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info("Checkpoint epoch: %s | val_loss: %.4f", ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))

    # ── Tokeniser + Dataset ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    test_df   = pd.read_csv(test_csv)
    logger.info("Test samples: %d", len(test_df))

    test_loader = DataLoader(
        IndianBiasDataset(test_df, tokenizer, max_len=cfg.get("max_len", _MAX_LEN)),
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            fc     = batch["fc_score"].to(device)
            labels = batch["bias_label"]

            logits = model(ids, mask, fc)["bias"]
            preds  = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CREDIBLEX v2 — INDIAN BIAS EVALUATION REPORT")
    print("=" * 65)

    report = classification_report(
        all_labels, all_preds,
        target_names=_TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(all_labels, all_preds)

    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print("=" * 65 + "\n")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    _save_confusion_matrix(all_labels, all_preds, output_png)
    logger.info("Confusion matrix saved → %s", output_png)


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    """
    Generate and save a seaborn confusion matrix heatmap.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        output_path: File path for the saved PNG image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    # Normalise row-wise to show recall percentages
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for empty classes
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=cm,           # show raw counts in cells
        fmt="d",
        cmap="Blues",
        xticklabels=_TARGET_NAMES,
        yticklabels=_TARGET_NAMES,
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("CredibleX v2 — Indian Bias Confusion Matrix", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CredibleX v2 Indian bias model.")
    parser.add_argument("--checkpoint", "-c", default=_DEFAULT_CKPT, help="Path to .pth checkpoint.")
    parser.add_argument("--test-csv",   "-t", default=_DEFAULT_CSV,  help="Path to test split CSV.")
    parser.add_argument("--output",     "-o", default=_DEFAULT_OUT,  help="Output PNG path for confusion matrix.")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.test_csv, args.output)
