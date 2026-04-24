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

        enc = self.tokenizer(
            text,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bias_label":     torch.tensor(int(row["bias_label"]),    dtype=torch.long),
            "fact_score":     torch.tensor(float(row["fact_score"]),  dtype=torch.float),
            "intent_label":   torch.tensor(int(row["intent_label"]),  dtype=torch.long),
            "emotion_label":  torch.tensor(int(row["emotion_label"]), dtype=torch.long),
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
            enc    = tokenizer(texts, max_length=config.MAX_LEN,
                               padding="longest", truncation=True,
                               return_tensors="pt")
            ids    = enc["input_ids"].to(config.DEVICE)
            mask   = enc["attention_mask"].to(config.DEVICE)
            try:
                out    = model(ids, mask)
                preds  = torch.argmax(out["bias"], dim=1).cpu().tolist()
                labels = batch["bias_label"].astype(int).tolist()
                correct += sum(p == l for p, l in zip(preds, labels))
            except RuntimeError:
                pass
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
        (df["emotion_label"] == 27).mean() * 100))

    # ── 3. Tokenizer & model ──────────────────────────────────────────────────
    print("\nLoading tokenizer: {}".format(config.MODEL_NAME))
    TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    dataset    = NewsDataset(df, TOKENIZER)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,        # Windows: must be 0 outside __main__ guard
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print("Loading model: {}".format(config.MODEL_NAME))
    model = NewsTrustModel(config.MODEL_NAME).to(device)

    # torch.compile speedup (PyTorch >= 2.0)
    # Disabled on Windows because Triton is not natively supported and crashes during forward pass
    # if hasattr(torch, "compile") and os.name != "nt":
    #     try:
    #         model = torch.compile(model)
    #         print("torch.compile() applied (~15% speedup)")
    #     except Exception as e:
    #         print("torch.compile skipped: {}".format(e))

    # ── 4. Loss functions with class weights ──────────────────────────────────
    bias_w    = compute_class_weights(df["bias_label"],    config.N_BIAS_CLASSES, device)
    intent_w  = compute_class_weights(df["intent_label"],  3,                     device)
    emotion_w = compute_class_weights(df["emotion_label"], 28,                    device)

    print("\nClass weights:")
    print("  bias   : {}".format([round(w, 3) for w in bias_w.tolist()]))
    top5 = sorted(enumerate(emotion_w.tolist()), key=lambda x: x[1], reverse=True)[:5]
    print("  emotion (top-5): {}".format([(i, round(w, 2)) for i, w in top5]))

    loss_bias    = nn.CrossEntropyLoss(weight=bias_w,    label_smoothing=0.1)
    loss_intent  = nn.CrossEntropyLoss(weight=intent_w)
    loss_emotion = nn.CrossEntropyLoss(weight=emotion_w)
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
    print("  Loss weights: bias x{} | emotion x{} | fact/intent x1.0".format(
        config.BIAS_LOSS_WEIGHT, config.EMOTION_LOSS_WEIGHT))

    # ── 6. Auto-resume ────────────────────────────────────────────────────────
    best_bias_acc = 0.0
    best_path     = os.path.join(config.CHECKPOINT_DIR, "best_bias_acc.pth")
    ckpt_path, start_epoch = _find_latest_checkpoint()

    if ckpt_path:
        print("\nResuming from epoch {} checkpoint: {}".format(
            start_epoch, ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device)
        # Handle torch.compile wrapping
        state = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Strip _orig_mod prefix if compiled
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
            ids    = batch["input_ids"].to(device, non_blocking=True)
            mask   = batch["attention_mask"].to(device, non_blocking=True)
            b_lbl  = batch["bias_label"].to(device, non_blocking=True)
            f_lbl  = batch["fact_score"].to(device, non_blocking=True)
            i_lbl  = batch["intent_label"].to(device, non_blocking=True)
            e_lbl  = batch["emotion_label"].to(device, non_blocking=True)

            # Forward + loss
            try:
                if scaler:
                    with torch.cuda.amp.autocast():
                        out  = model(ids, mask)
                        lb   = loss_bias(out["bias"], b_lbl)
                        lf   = loss_fact(out["factuality"].squeeze(-1), f_lbl)
                        li   = loss_intent(out["intent"], i_lbl)
                        le   = loss_emotion(out["emotion"], e_lbl)
                        loss = (config.BIAS_LOSS_WEIGHT * lb + lf + li +
                                config.EMOTION_LOSS_WEIGHT * le) / config.GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                else:
                    out  = model(ids, mask)
                    lb   = loss_bias(out["bias"], b_lbl)
                    lf   = loss_fact(out["factuality"].squeeze(-1), f_lbl)
                    li   = loss_intent(out["intent"], i_lbl)
                    le   = loss_emotion(out["emotion"], e_lbl)
                    loss = (config.BIAS_LOSS_WEIGHT * lb + lf + li +
                            config.EMOTION_LOSS_WEIGHT * le) / config.GRAD_ACCUM_STEPS
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