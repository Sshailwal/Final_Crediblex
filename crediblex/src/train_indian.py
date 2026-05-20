"""
Training script for CredibleX v2 — Indian Political Bias Classifier.

Usage:
    python src/train_indian.py
    python src/train_indian.py --config configs/indian_finetune.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import CredibleXv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_DEFAULTS: dict = {
    "model_name": "xlm-roberta-base",
    "epochs": 5,
    "batch_size": 16,
    "lr_encoder": 5e-5,
    "lr_head": 2e-4,
    "max_len": 256,
    "warmup_steps": 100,
    "device": "auto",
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "checkpoint_dir": "checkpoints",
    "checkpoint_name": "model_v2_indian.pth",
}

_LABEL_NAMES = ["Far Left", "Left", "Center", "Right", "Far Right"]


class IndianBiasDataset(Dataset):
    """
    PyTorch Dataset for the Indian political bias corpus.

    Each item yields input_ids, attention_mask, fc_score (default 0.5),
    and bias_label.

    Args:
        df:        DataFrame with ``text``, ``label``, and optionally ``fc_score``.
        tokenizer: HuggingFace tokeniser compatible with the chosen backbone.
        max_len:   Maximum token sequence length.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256) -> None:
        self.texts     = df["text"].tolist()
        self.labels    = df["label"].tolist()
        self.fc_scores = df["fc_score"].tolist() if "fc_score" in df.columns else [0.5] * len(df)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "fc_score":       torch.tensor(float(self.fc_scores[idx]), dtype=torch.float),
            "bias_label":     torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def compute_class_weights(df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """
    Balanced class weights: total / (num_classes * per_class_count).

    Args:
        df:          DataFrame with a ``label`` column.
        num_classes: Number of output classes.

    Returns:
        Float tensor of shape [num_classes].
    """
    total = len(df)
    weights = []
    for cls in range(num_classes):
        count = (df["label"] == cls).sum()
        w = total / (num_classes * count) if count > 0 else 1.0
        if count == 0:
            logger.warning("Class %d (%s) has 0 samples — weight set to 1.0.", cls, _LABEL_NAMES[cls])
        weights.append(w)
    logger.info("Class weights: %s", {_LABEL_NAMES[i]: round(w, 3) for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32)


def train(cfg: dict) -> None:
    """
    Full training loop for CredibleXv2.

    Trains for cfg['epochs'] epochs with class-weighted CrossEntropyLoss,
    cosine schedule with linear warmup, and saves best val-loss checkpoint.

    Args:
        cfg: Configuration dictionary (see _DEFAULTS for valid keys).

    Raises:
        FileNotFoundError: If train/val CSVs are missing.
        RuntimeError:      On CUDA OOM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg["device"] == "auto" else torch.device(cfg["device"])
    logger.info("Device: %s", device)

    for p in (cfg["train_csv"], cfg["val_csv"]):
        if not Path(p).exists():
            raise FileNotFoundError(f"Data file not found: {p}. Run `python src/data_prep.py` first.")

    train_df = pd.read_csv(cfg["train_csv"])
    val_df   = pd.read_csv(cfg["val_csv"])
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_loader = DataLoader(
        IndianBiasDataset(train_df, tokenizer, cfg["max_len"]),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        IndianBiasDataset(val_df, tokenizer, cfg["max_len"]),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CredibleXv2(backbone=cfg["model_name"]).to(device)

    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params    = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer = AdamW(
        [{"params": encoder_params, "lr": float(cfg["lr_encoder"])},
         {"params": head_params,    "lr": float(cfg["lr_head"])}],
        weight_decay=1e-2,
    )

    total_steps = len(train_loader) * cfg["epochs"]
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_steps"]),
        num_training_steps=total_steps,
    )

    class_weights = compute_class_weights(train_df).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_dir  = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["checkpoint_name"]

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]", leave=False):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            fc     = batch["fc_score"].to(device)
            labels = batch["bias_label"].to(device)

            optimizer.zero_grad()
            try:
                logits = model(ids, mask, fc)["bias"]
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.error("CUDA OOM — reduce batch_size in config.")
                raise
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)

        avg_t = t_loss / len(train_loader)
        acc_t = t_correct / t_total

        # ── Val ──────────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                fc     = batch["fc_score"].to(device)
                labels = batch["bias_label"].to(device)
                logits = model(ids, mask, fc)["bias"]
                v_loss    += criterion(logits, labels).item()
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        avg_v = v_loss / len(val_loader)
        acc_v = v_correct / v_total
        history.append({"epoch": epoch, "train_loss": round(avg_t, 4),
                         "val_loss": round(avg_v, 4), "train_acc": round(acc_t, 4),
                         "val_acc": round(acc_v, 4)})
        logger.info("Epoch %d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f",
                    epoch, avg_t, acc_t, avg_v, acc_v)

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": avg_v, "val_acc": acc_v, "config": cfg}, ckpt_path)
            logger.info("  ✓ Best checkpoint saved → %s", ckpt_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"\n  {'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc'}")
    print("  " + "─" * 52)
    for r in history:
        print(f"  {r['epoch']:<6} {r['train_loss']:<12} {r['train_acc']:<12} {r['val_loss']:<12} {r['val_acc']}")


def _load_config(yaml_path: Optional[str]) -> dict:
    """Load YAML config and merge with defaults."""
    cfg = dict(_DEFAULTS)
    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CredibleX v2 Indian bias model.")
    parser.add_argument("--config", "-c", default="configs/indian_finetune.yaml")
    args = parser.parse_args()
    train(_load_config(args.config))
