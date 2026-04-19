"""
train.py — CredibleX Training Script (Overnight Edition)
=========================================================
Features
--------
• Gradient clipping (max_norm=1.0) — prevents exploding gradients in DeBERTa fine-tuning
• Gradient accumulation (GRAD_ACCUM_STEPS) — effective batch size without extra VRAM
• Linear LR warmup + decay (get_linear_schedule_with_warmup)
• Class-weighted CrossEntropyLoss — fixes Left/Right/Center imbalance
• Auto-resume from latest epoch checkpoint — crash = restart from last save
• Per-epoch checkpoint (model + optimizer + scheduler + scaler state)
• Per-batch ETA display (updates every 10 batches)
• Hourly full-status block printed to console
• VRAM monitoring after every epoch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import os
import sys
import glob
import time
from datetime import datetime
from functools import partial

# Force UTF-8 so print statements work on Windows cp1252 consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from data_ingest import get_explainable_dataset
from model import NewsTrustModel
import config

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ── Module-level tokenizer (required for Windows multiprocessing pickling) ────
TOKENIZER = None


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Format raw seconds into a human-readable string."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def find_latest_checkpoint() -> tuple:
    """
    Scan CHECKPOINT_DIR for saved epoch checkpoints.
    Returns (path, epoch_number) of the latest, or (None, 0) if none found.
    """
    pattern = os.path.join(config.CHECKPOINT_DIR, "epoch_*.pth")
    found   = glob.glob(pattern)
    if not found:
        return None, 0

    def epoch_num(p):
        try:
            return int(os.path.basename(p).replace("epoch_", "").replace(".pth", ""))
        except Exception:
            return 0

    found.sort(key=epoch_num)
    latest = found[-1]
    return latest, epoch_num(latest)


def compute_class_weights(series: pd.Series, n_classes: int,
                           device: str) -> torch.Tensor:
    """Inverse-frequency weights → minority classes get larger loss weight."""
    counts  = series.value_counts().to_dict()
    total   = len(series)
    weights = [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df      = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text":          str(row["text"]),
            "bias_label":    torch.tensor(int(row["bias_label"]),   dtype=torch.long),
            "fact_label":    torch.tensor(float(row["fact_score"]), dtype=torch.float),
            "intent_label":  torch.tensor(int(row["intent_label"]), dtype=torch.long),
            "emotion_label": torch.tensor(int(row["emotion_label"]),dtype=torch.long),
        }


def collate_fn(batch, max_len=512):
    if TOKENIZER is None:
        raise RuntimeError("TOKENIZER not initialised. Set before creating DataLoader.")
    texts    = [item["text"] for item in batch]
    encoding = TOKENIZER(texts, max_length=max_len, padding="longest",
                         truncation=True, return_tensors="pt")
    return {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bias_label":     torch.stack([x["bias_label"]    for x in batch]).long(),
        "fact_label":     torch.stack([x["fact_label"]    for x in batch]).float(),
        "intent_label":   torch.stack([x["intent_label"]  for x in batch]).long(),
        "emotion_label":  torch.stack([x["emotion_label"] for x in batch]).long(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── 1. Load / generate dataset ────────────────────────────────────────────
    csv_path = "training_data.csv"
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} rows from {csv_path}")
    else:
        print("⚠️  training_data.csv not found — generating now (may take 10–20 min)...")
        df = get_explainable_dataset()
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(df):,} rows → {csv_path}")

    print(f"\n📊 Bias  : {df['bias_label'].value_counts().to_dict()}")
    print(f"📊 Intent: {df['intent_label'].value_counts().to_dict()}")

    # ── 2. Tokenizer & DataLoader ─────────────────────────────────────────────
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
    dataset   = NewsDataset(df, TOKENIZER, config.MAX_LEN)
    loader    = DataLoader(
        dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (config.DEVICE == "cuda"),
        collate_fn  = partial(collate_fn, max_len=config.MAX_LEN),
    )
    print(f"\n📦 {len(loader)} batches/epoch | batch={config.BATCH_SIZE} | "
          f"grad_accum={config.GRAD_ACCUM_STEPS} | eff_batch={config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = NewsTrustModel(config.MODEL_NAME, dropout=config.DROPOUT).to(config.DEVICE)
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB | "
              f"Used after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── 4. Loss functions with class weights ──────────────────────────────────
    bias_w   = compute_class_weights(df["bias_label"],   3, config.DEVICE)
    intent_w = compute_class_weights(df["intent_label"], 3, config.DEVICE)
    print(f"\n⚖️  Bias weights  : {[round(w,3) for w in bias_w.tolist()]}  (Left/Center/Right)")
    print(f"⚖️  Intent weights: {[round(w,3) for w in intent_w.tolist()]}  (News/Opinion/Satire)")

    loss_bias    = nn.CrossEntropyLoss(weight=bias_w)
    loss_intent  = nn.CrossEntropyLoss(weight=intent_w)
    loss_emotion = nn.CrossEntropyLoss()
    loss_fact    = nn.MSELoss()

    # ── 5. Optimizer + LR scheduler ───────────────────────────────────────────
    optimizer        = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    total_opt_steps  = (len(loader) // config.GRAD_ACCUM_STEPS) * config.EPOCHS
    warmup_steps     = max(1, int(total_opt_steps * config.WARMUP_RATIO))
    scheduler        = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
    scaler           = torch.amp.GradScaler("cuda") if config.DEVICE == "cuda" else None

    print(f"\n🗓️  {config.EPOCHS} epochs | {total_opt_steps} opt-steps | {warmup_steps} warmup-steps")

    # ── 6. Auto-resume from latest checkpoint ────────────────────────────────
    ckpt_path, start_epoch = find_latest_checkpoint()
    if ckpt_path:
        print(f"\n🔄 Checkpoint found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"   ✅ Resumed from epoch {start_epoch}. Starting epoch {start_epoch + 1}.")
    else:
        print("\n🆕 No checkpoint found — starting fresh training.")

    if start_epoch >= config.EPOCHS:
        print(f"✅ All {config.EPOCHS} epochs already complete. Nothing to do.")
        return

    # ── 7. Training loop ──────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"🚀 TRAINING STARTED — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*74}\n")

    train_start   = time.time()
    last_hourly   = train_start
    HOURLY_SEC    = 3600

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            global_batch  = epoch * len(loader) + batch_idx
            total_batches = config.EPOCHS * len(loader)

            # Move to device
            ids   = batch["input_ids"].to(config.DEVICE)
            mask  = batch["attention_mask"].to(config.DEVICE)
            b_lbl = batch["bias_label"].to(config.DEVICE)
            f_lbl = batch["fact_label"].to(config.DEVICE)
            i_lbl = batch["intent_label"].to(config.DEVICE)
            e_lbl = batch["emotion_label"].to(config.DEVICE)

            # ── Forward + backward ────────────────────────────────────────────
            if config.DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    out  = model(ids, mask)
                    lb   = loss_bias(out["bias"], b_lbl)
                    lf   = loss_fact(out["factuality"].squeeze(-1), f_lbl)
                    li   = loss_intent(out["intent"], i_lbl)
                    le   = loss_emotion(out["emotion"], e_lbl)
                    loss = (lb + lf + li + le) / config.GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                out  = model(ids, mask)
                lb   = loss_bias(out["bias"], b_lbl)
                lf   = loss_fact(out["factuality"].squeeze(-1), f_lbl)
                li   = loss_intent(out["intent"], i_lbl)
                le   = loss_emotion(out["emotion"], e_lbl)
                loss = (lb + lf + li + le) / config.GRAD_ACCUM_STEPS
                loss.backward()

            epoch_loss += loss.item() * config.GRAD_ACCUM_STEPS  # un-scale for display

            # ── Optimizer step (with grad clipping) every GRAD_ACCUM_STEPS ───
            last_batch = batch_idx == len(loader) - 1
            if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0 or last_batch:
                if config.DEVICE == "cuda":
                    scaler.unscale_(optimizer)
                # GRADIENT CLIPPING — critical for DeBERTa fine-tuning stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
                if config.DEVICE == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── Progress line (every 10 batches) ─────────────────────────────
            if batch_idx % 10 == 0 or last_batch:
                elapsed = time.time() - train_start
                done    = global_batch + 1
                eta     = (elapsed / done) * (total_batches - done) if done else 0
                avg_l   = epoch_loss / (batch_idx + 1)
                gpu_str = ""
                if config.DEVICE == "cuda":
                    gm = torch.cuda.memory_allocated() / 1e9
                    gpu_str = f" | 💾 {gm:.2f}GB"
                print(
                    f"\r  [Ep {epoch+1}/{config.EPOCHS} | Bt {batch_idx+1}/{len(loader)} | "
                    f"Loss {avg_l:.4f} | ⏱ {fmt_time(elapsed)} | ETA {fmt_time(eta)}{gpu_str}]   ",
                    end="", flush=True,
                )

            # ── Hourly status block ───────────────────────────────────────────
            if time.time() - last_hourly >= HOURLY_SEC:
                elapsed = time.time() - train_start
                done    = global_batch + 1
                eta     = (elapsed / done) * (total_batches - done) if done else 0
                avg_l   = epoch_loss / (batch_idx + 1)
                print(f"\n\n{'='*74}")
                print(f"⏰  HOURLY STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Epoch        : {epoch+1} / {config.EPOCHS}")
                print(f"   Batch        : {batch_idx+1} / {len(loader)}")
                print(f"   Avg Loss     : {avg_l:.4f}")
                print(f"   Time Elapsed : {fmt_time(elapsed)}")
                print(f"   ETA          : {fmt_time(eta)}")
                if config.DEVICE == "cuda":
                    gm = torch.cuda.memory_allocated() / 1e9
                    vm = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"   GPU Memory   : {gm:.2f} GB / {vm:.1f} GB")
                print(f"{'='*74}\n")
                last_hourly = time.time()

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss = epoch_loss / len(loader)
        print(f"\n\n✅ Epoch {epoch+1}/{config.EPOCHS} COMPLETE | "
              f"Avg Loss: {avg_loss:.4f} | {datetime.now().strftime('%H:%M:%S')}")

        # Save epoch checkpoint (model + optimizer + scheduler + scaler state)
        ckpt_save = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        save_dict = {
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss":                 avg_loss,
        }
        if scaler:
            save_dict["scaler_state_dict"] = scaler.state_dict()
        torch.save(save_dict, ckpt_save)
        print(f"💾 Checkpoint saved → {ckpt_save}")

        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()
            gm = torch.cuda.memory_allocated() / 1e9
            print(f"   GPU mem after cache clear: {gm:.2f} GB")

    # ── Final model save ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), config.SAVE_PATH)
    total_time = time.time() - train_start
    print(f"\n{'='*74}")
    print(f"🎉 TRAINING COMPLETE in {fmt_time(total_time)}")
    print(f"   Final model → {config.SAVE_PATH}")
    print(f"   Run  python evaluate.py  to check accuracy metrics.")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    train()