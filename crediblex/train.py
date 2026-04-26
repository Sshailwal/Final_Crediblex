# emotion_label: int→multihot
# coding: utf-8
"""
train.py - CredibleX multi-task trainer (DeBERTa-v3-base)
Targets: RTX 4050 6 GB VRAM | BATCH=8 | MAX_LEN=256 | GradAccum=4
"""

import os, sys, glob, time, argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import NewsTrustModel
import config

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:
        return "{}h {}m {}s".format(h, m, s)
    if m:
        return "{}m {}s".format(m, s)
    return "{}s".format(s)


def _safe_save(path, obj):
    """Write to .tmp then atomically rename — safe on power cut."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _find_latest_checkpoint():
    """Return (path, epoch_num) of the highest epoch_N.pth, or (None, 0)."""
    pattern = os.path.join(config.CHECKPOINT_DIR, "epoch_*.pth")
    files   = glob.glob(pattern)
    if not files:
        mid = os.path.join(config.CHECKPOINT_DIR, "mid_epoch.pth")
        if os.path.isfile(mid):
            print("  Mid-epoch checkpoint found — will load weights only.")
            return mid, 0
        return None, 0

    def _epoch_num(p):
        try:
            return int(os.path.basename(p)
                       .replace("epoch_", "").replace(".pth", ""))
        except Exception:
            return 0

    files.sort(key=_epoch_num)
    latest = files[-1]
    return latest, _epoch_num(latest)


def compute_class_weights(series, n_classes, device):
    counts = series.value_counts()
    total  = len(series)
    w = []
    for c in range(n_classes):
        cnt = counts.get(c, 1)
        w.append(total / (n_classes * cnt))
    return torch.tensor(w, dtype=torch.float32, device=device)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row["text"])

        # Bug 11: emotion_label is now a JSON multi-hot string
        emotion_vec = torch.tensor(json.loads(row["emotion_label"]), dtype=torch.float32)
        
        w_ids, w_masks = NewsTrustModel.create_sliding_windows(text, self.tokenizer, config.MAX_LEN)

        return {
            "windows_ids":    w_ids,
            "windows_masks":  w_masks,
            "bias_label":     torch.tensor(int(row["bias_label"]),    dtype=torch.long),
            "fact_score":     torch.tensor(float(row["fact_score"]),  dtype=torch.float),
            "intent_label":   torch.tensor(int(row["intent_label"]),  dtype=torch.long),
            "emotion_label":  emotion_vec,
        }

def custom_collate_fn(batch):
    return {
        "windows_ids": [item["windows_ids"] for item in batch],
        "windows_masks": [item["windows_masks"] for item in batch],
        "bias_label": torch.stack([item["bias_label"] for item in batch]),
        "fact_score": torch.stack([item["fact_score"] for item in batch]),
        "intent_label": torch.stack([item["intent_label"] for item in batch]),
        "emotion_label": torch.stack([item["emotion_label"] for item in batch]),
    }


# ── Bias accuracy on random sample ───────────────────────────────────────────

def _quick_bias_acc(model, df, tokenizer, n=512):
    sample = df.sample(n=min(n, len(df)), random_state=99).reset_index(drop=True)
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(sample), config.BATCH_SIZE):
            batch  = sample.iloc[i : i + config.BATCH_SIZE]
            texts  = batch["text"].astype(str).tolist()
            labels = batch["bias_label"].astype(int).tolist()
            
            for text, label in zip(texts, labels):
                w_ids, w_masks = NewsTrustModel.create_sliding_windows(text, tokenizer, config.MAX_LEN)
                
                pooled_embs = []
                for wid, wmask in zip(w_ids, w_masks):
                    wid = wid.unsqueeze(0).to(config.DEVICE)
                    wmask = wmask.unsqueeze(0).to(config.DEVICE)
                    out = model.backbone(input_ids=wid, attention_mask=wmask)
                    pooled = model._mean_pool(out.last_hidden_state, wmask)
                    pooled_embs.append(pooled)
                    
                avg_pooled = torch.stack(pooled_embs).mean(dim=0)
                pred_bias = model.bias_head(avg_pooled)
                pred_idx = torch.argmax(pred_bias, dim=1).item()
                if pred_idx == label:
                    correct += 1
    model.train()
    return correct / len(sample)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    # ── 0. CLI ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="CredibleX Trainer")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete all checkpoints and start fresh")
    args, _ = parser.parse_known_args()

    if args.fresh:
        ckpts = (glob.glob(os.path.join(config.CHECKPOINT_DIR, "*.pth")) +
                 glob.glob(os.path.join(config.CHECKPOINT_DIR, "*.tmp")))
        if ckpts:
            ans = input(
                "  Delete {:d} checkpoint file(s) in {}? [y/N]: ".format(
                    len(ckpts), config.CHECKPOINT_DIR))
            if ans.strip().lower() == "y":
                for f in ckpts:
                    os.remove(f)
                print("  Checkpoints deleted. Starting fresh.\n")
            else:
                print("  Aborted.")
                sys.exit(0)

    # ── 1. CUDA info ──────────────────────────────────────────────────────────
    device = config.DEVICE
    print("\n" + "=" * 60)
    print("  CredibleX Trainer")
    print("  Device: {}".format(device))
    if device.type == "cuda":
        print("  GPU   : {}".format(torch.cuda.get_device_name(0)))
        print("  VRAM  : {:.1f} GB total".format(
            torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("  BATCH_SIZE={} | MAX_LEN={} | GRAD_ACCUM={}".format(
        config.BATCH_SIZE, config.MAX_LEN, config.GRAD_ACCUM_STEPS))
    print("=" * 60 + "\n")

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    csv_path = "training_data.csv"
    if not os.path.isfile(csv_path):
        print("ERROR: training_data.csv not found.")
        print("  Run: python prepare_dataset.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print("Loaded {:,} rows from {}".format(len(df), csv_path))
    print("  bias  distribution: {}".format(
        df["bias_label"].value_counts().sort_index().to_dict()))
    print("  emotion neutral %: {:.1f}".format(
        df["emotion_label"].apply(lambda s: json.loads(s)[27] == 1.0).mean() * 100))

    # ── 3. Tokenizer & model ──────────────────────────────────────────────────
    print("\nLoading tokenizer: {}".format(config.MODEL_NAME))
    TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)

    dataset    = NewsDataset(df, TOKENIZER)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,        # Windows: must be 0 outside __main__ guard
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    print("Loading model: {}".format(config.MODEL_NAME))
    model = NewsTrustModel(config.MODEL_NAME).to(device)

    # ── 4. Loss functions with class weights ──────────────────────────────────
    bias_w    = compute_class_weights(df["bias_label"],    config.N_BIAS_CLASSES, device)
    intent_w  = compute_class_weights(df["intent_label"],  3,                     device)

    print("\nClass weights:")
    print("  bias   : {}".format([round(w, 3) for w in bias_w.tolist()]))

    loss_bias    = nn.CrossEntropyLoss(weight=bias_w,    label_smoothing=0.1)
    loss_intent  = nn.CrossEntropyLoss(weight=intent_w)
    # Bug 11: Multi-label emotion uses BCEWithLogitsLoss
    loss_emotion = nn.BCEWithLogitsLoss()
    loss_fact    = nn.MSELoss()

    # ── 5. Optimizer + scheduler ──────────────────────────────────────────────
    n_batches        = len(dataloader)
    total_opt_steps  = (n_batches // config.GRAD_ACCUM_STEPS) * config.EPOCHS
    warmup_steps     = int(total_opt_steps * config.WARMUP_RATIO)

    optimizer  = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                       weight_decay=0.01)
    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print("\nTraining plan:")
    print("  {} epochs | {} batches/epoch | {} opt-steps | {} warmup".format(
        config.EPOCHS, n_batches, total_opt_steps, warmup_steps))
    print("  Loss weights: bias x{} | fact x{} | emotion x{} | intent x1.0".format(
        config.BIAS_LOSS_WEIGHT, config.FACT_LOSS_WEIGHT, config.EMOTION_LOSS_WEIGHT))

    # ── 6. Auto-resume ────────────────────────────────────────────────────────
    best_bias_acc = 0.0
    best_path     = os.path.join(config.CHECKPOINT_DIR, "best_bias_acc.pth")
    ckpt_path, start_epoch = _find_latest_checkpoint()

    if ckpt_path:
        print("\nResuming from epoch {} checkpoint: {}".format(
            start_epoch, ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            new_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_bias_acc = ckpt.get("best_bias_acc", 0.0)
        print("  best_bias_acc so far: {:.3f}".format(best_bias_acc))
    else:
        start_epoch = 0
        print("\nNo checkpoint found — training from scratch.")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    train_start     = time.time()
    epoch_durations = []

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc="Epoch {}/{}".format(
            epoch + 1, config.EPOCHS), unit="batch", dynamic_ncols=True)

        torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(pbar):
            last_batch = (batch_idx == n_batches - 1)

            # Move to device
            b_lbl  = batch["bias_label"].to(device, non_blocking=True)
            f_lbl  = batch["fact_score"].to(device, non_blocking=True)
            i_lbl  = batch["intent_label"].to(device, non_blocking=True)
            # Bug 11: emotion labels are float vectors for BCE
            e_lbl  = batch["emotion_label"].to(device, non_blocking=True)

            try:
                batch_bias_preds = []
                batch_fact_preds = []
                batch_intent_preds = []
                batch_emotion_preds = []

                for i in range(len(batch["windows_ids"])):
                    w_ids_list = batch["windows_ids"][i]
                    w_masks_list = batch["windows_masks"][i]
                    
                    pooled_embs = []
                    for w_ids, w_mask in zip(w_ids_list, w_masks_list):
                        w_ids = w_ids.unsqueeze(0).to(device, non_blocking=True)
                        w_mask = w_mask.unsqueeze(0).to(device, non_blocking=True)
                        
                        if scaler:
                            with torch.cuda.amp.autocast():
                                out = model.backbone(input_ids=w_ids, attention_mask=w_mask)
                                pooled = model._mean_pool(out.last_hidden_state, w_mask)
                        else:
                            out = model.backbone(input_ids=w_ids, attention_mask=w_mask)
                            pooled = model._mean_pool(out.last_hidden_state, w_mask)
                        
                        pooled_embs.append(pooled)
                    
                    avg_pooled = torch.stack(pooled_embs).mean(dim=0)
                    avg_pooled = model.dropout(avg_pooled)
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            batch_bias_preds.append(model.bias_head(avg_pooled))
                            batch_fact_preds.append(torch.sigmoid(model.fact_head(avg_pooled)))
                            batch_intent_preds.append(model.intent_head(avg_pooled))
                            batch_emotion_preds.append(model.emotion_head(avg_pooled))
                    else:
                        batch_bias_preds.append(model.bias_head(avg_pooled))
                        batch_fact_preds.append(torch.sigmoid(model.fact_head(avg_pooled)))
                        batch_intent_preds.append(model.intent_head(avg_pooled))
                        batch_emotion_preds.append(model.emotion_head(avg_pooled))
                        
                out_bias = torch.cat(batch_bias_preds, dim=0)
                out_fact = torch.cat(batch_fact_preds, dim=0).squeeze(-1)
                out_intent = torch.cat(batch_intent_preds, dim=0)
                out_emotion = torch.cat(batch_emotion_preds, dim=0)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        lb = loss_bias(out_bias, b_lbl)
                        lf = loss_fact(out_fact, f_lbl)
                        li = loss_intent(out_intent, i_lbl)
                        le = loss_emotion(out_emotion, e_lbl)
                        loss = (config.BIAS_LOSS_WEIGHT * lb + config.FACT_LOSS_WEIGHT * lf + li + config.EMOTION_LOSS_WEIGHT * le) / config.GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                else:
                    lb = loss_bias(out_bias, b_lbl)
                    lf = loss_fact(out_fact, f_lbl)
                    li = loss_intent(out_intent, i_lbl)
                    le = loss_emotion(out_emotion, e_lbl)
                    loss = (config.BIAS_LOSS_WEIGHT * lb + config.FACT_LOSS_WEIGHT * lf + li + config.EMOTION_LOSS_WEIGHT * le) / config.GRAD_ACCUM_STEPS
                    loss.backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print("\n[OOM] Skipping batch {}. Cleared VRAM cache.".format(batch_idx))
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            epoch_loss += loss.item() * config.GRAD_ACCUM_STEPS

            # VRAM log after first batch
            if batch_idx == 0 and device.type == "cuda":
                used = torch.cuda.memory_allocated() / 1e9
                tqdm.write("  VRAM used: {:.2f} GB / {:.1f} GB".format(
                    used,
                    torch.cuda.get_device_properties(0).total_memory / 1e9))

            # Optimizer step
            if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0 or last_batch:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Mid-epoch checkpoint every MID_EPOCH_CKPT_FREQ batches
            if (batch_idx + 1) % config.MID_EPOCH_CKPT_FREQ == 0:
                mid_path = os.path.join(config.CHECKPOINT_DIR, "mid_epoch.pth")
                mid_dict = {
                    "epoch":                epoch,
                    "batch_idx":            batch_idx,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_bias_acc":        best_bias_acc,
                }
                if scaler:
                    mid_dict["scaler_state_dict"] = scaler.state_dict()
                _safe_save(mid_path, mid_dict)

                epoch_elapsed  = time.time() - epoch_start
                batches_done   = batch_idx + 1
                secs_per_batch = epoch_elapsed / batches_done
                remaining_secs = (n_batches - batches_done) * secs_per_batch
                avg_loss_so_far = epoch_loss / batches_done
                tqdm.write(
                    "  [mid-ckpt] batch {}/{} | loss: {:.4f} | ~{} remaining this epoch".format(
                        batches_done, n_batches,
                        avg_loss_so_far,
                        _fmt_time(remaining_secs)))

            # tqdm bar update
            avg = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": "{:.4f}".format(avg)})

        pbar.close()

        # ── Epoch summary ──────────────────────────────────────────────────
        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_loss = epoch_loss / n_batches

        total_elapsed    = time.time() - train_start
        epochs_done      = epoch + 1 - start_epoch
        avg_epoch_time   = total_elapsed / epochs_done
        epochs_remaining = config.EPOCHS - (epoch + 1)
        eta_secs         = avg_epoch_time * epochs_remaining

        # Bias accuracy sample
        bias_acc = _quick_bias_acc(model, df, TOKENIZER)

        tqdm.write(
            "\nEpoch {}/{} complete | loss: {:.4f} | bias_acc: {:.1f}% | "
            "elapsed: {} | ETA: {}".format(
                epoch + 1, config.EPOCHS,
                avg_loss,
                bias_acc * 100,
                _fmt_time(total_elapsed),
                _fmt_time(eta_secs)))

        # Save epoch checkpoint (safe write)
        ckpt_save = os.path.join(config.CHECKPOINT_DIR,
                                 "epoch_{}.pth".format(epoch + 1))
        save_dict = {
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss":                 avg_loss,
            "best_bias_acc":        best_bias_acc,
            "config": {
                "N_BIAS_CLASSES":    config.N_BIAS_CLASSES,
                "BATCH_SIZE":        config.BATCH_SIZE,
                "MAX_LEN":           config.MAX_LEN,
                "GRAD_ACCUM_STEPS":  config.GRAD_ACCUM_STEPS,
                "EPOCHS":            config.EPOCHS,
            },
        }
        if scaler:
            save_dict["scaler_state_dict"] = scaler.state_dict()
        _safe_save(ckpt_save, save_dict)
        tqdm.write("  Checkpoint saved -> {}".format(ckpt_save))

        # Remove mid-epoch checkpoint once full epoch is saved
        mid_path = os.path.join(config.CHECKPOINT_DIR, "mid_epoch.pth")
        if os.path.isfile(mid_path):
            os.remove(mid_path)

        # Best model tracker
        if bias_acc > best_bias_acc:
            best_bias_acc = bias_acc
            save_dict["best_bias_acc"] = best_bias_acc
            _safe_save(best_path, save_dict)
            tqdm.write("  New best bias_acc: {:.1f}% -> saved {}".format(
                best_bias_acc * 100, best_path))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Training complete ──────────────────────────────────────────────────────
    total_time = time.time() - train_start
    print("\n" + "=" * 60)
    print("  Training complete in {}".format(_fmt_time(total_time)))
    print("  Best bias accuracy: {:.1f}%".format(best_bias_acc * 100))
    print("  Best model        : {}".format(best_path))
    print("=" * 60)

    # Save final model
    _safe_save(config.SAVE_PATH, model.state_dict())
    print("  Final model saved -> {}".format(config.SAVE_PATH))


if __name__ == "__main__":
    train()