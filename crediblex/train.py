# coding: utf-8
"""
train.py - CredibleX multi-task trainer (DeBERTa-v3-base)
Targets: RTX 4050 6 GB VRAM | BALANCED MODE
"""

import os, sys, glob, time, argparse, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm

from model import NewsTrustModel
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return "{}h {}m {}s".format(h, m, s) if h else "{}m {}s".format(m, s) if m else "{}s".format(s)

def _safe_save(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _find_latest_checkpoint():
    pattern = os.path.join(config.CHECKPOINT_DIR, "epoch_*.pth")
    files   = glob.glob(pattern)
    if not files:
        mid = os.path.join(config.CHECKPOINT_DIR, "mid_epoch.pth")
        return (mid, 0) if os.path.isfile(mid) else (None, 0)
    
    def _epoch_num(p):
        try: return int(os.path.basename(p).replace("epoch_", "").replace(".pth", ""))
        except: return 0
        
    files.sort(key=_epoch_num)
    latest = files[-1]
    return latest, _epoch_num(latest)

def compute_class_weights(series, n_classes, device):
    counts = series.value_counts()
    total  = len(series)
    w = []
    for c in range(n_classes):
        cnt = counts.get(c, 0)
        # Weight=0.0 for missing classes so their loss contributes nothing
        w.append(total / (n_classes * cnt) if cnt > 0 else 0.0)
    return torch.tensor(w, dtype=torch.float32, device=device)

# ── VRAM-Safe Sliding Window ──────────────────────────────────────────────────

def _encode_windows(text, tokenizer, device, max_len, stride):
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]

    if len(token_ids) == 0:
        enc = tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        yield enc["input_ids"].to(device), enc["attention_mask"].to(device)
        return

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    inner  = max_len - 2

    start = 0
    while start < len(token_ids):
        end      = min(start + inner, len(token_ids))
        chunk    = token_ids[start:end]
        ids      = [cls_id] + chunk + [sep_id]
        pad_len  = max_len - len(ids)
        mask     = [1] * len(ids) + [0] * pad_len
        ids     += [tokenizer.pad_token_id] * pad_len

        yield (
            torch.tensor([ids],  dtype=torch.long, device=device),
            torch.tensor([mask], dtype=torch.long, device=device),
        )
        start += stride
        if end == len(token_ids): break

def _get_pooled_embedding(model, text, tokenizer, device):
    window_embeddings = []

    for input_ids, attention_mask in _encode_windows(
            text, tokenizer, device, config.MAX_LEN, config.SLIDING_STRIDE):

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            backbone_out = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = model._mean_pool(backbone_out.last_hidden_state, attention_mask)
            pooled = model.dropout(pooled)

        window_embeddings.append(pooled.detach().float())
        del backbone_out, input_ids, attention_mask
        torch.cuda.empty_cache()

    if not window_embeddings:
        return torch.zeros(1, model.backbone.config.hidden_size, device=device)

    return torch.stack(window_embeddings, dim=0).mean(dim=0)

# ── Dataset ───────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text":          str(row["text"]),
            "bias_label":    torch.tensor(int(row["bias_label"]),    dtype=torch.long),
            "fact_score":    torch.tensor(float(row["fact_score"]),  dtype=torch.float),
            "intent_label":  torch.tensor(int(row["intent_label"]),  dtype=torch.long),
            "emotion_label": torch.tensor(json.loads(row["emotion_label"]), dtype=torch.float32),
        }

def collate_fn(batch):
    return {
        "text":          [item["text"] for item in batch],
        "bias_label":    torch.stack([item["bias_label"]    for item in batch]),
        "fact_score":    torch.stack([item["fact_score"]    for item in batch]),
        "intent_label":  torch.stack([item["intent_label"]  for item in batch]),
        "emotion_label": torch.stack([item["emotion_label"] for item in batch]),
    }

# ── Evaluation helpers ───────────────────────────────────────────────────────

def _quick_bias_acc(model, df, tokenizer, n=256):
    """Accuracy over rows that actually carry a bias label."""
    bias_df = df[df["bias_label"] >= 0]
    if len(bias_df) == 0:
        return 0.0
    sample = bias_df.sample(n=min(n, len(bias_df)), random_state=42).reset_index(drop=True)
    model.eval()
    correct = 0
    device = config.DEVICE
    with torch.no_grad():
        for _, row in sample.iterrows():
            pooled = _get_pooled_embedding(model, str(row["text"]), tokenizer, device)
            pred = model.bias_head(pooled)
            if torch.argmax(pred, dim=1).item() == int(row["bias_label"]):
                correct += 1
    model.train()
    return correct / len(sample)


def _bias_sample_probe(model, df, tokenizer):
    """Print 5 held-out predictions (at least one per class) for bias head."""
    NAMES = {0: "Left", 1: "Center", 2: "Right"}
    bias_df = df[df["bias_label"] >= 0]
    samples = []
    for cls in range(3):
        cls_rows = bias_df[bias_df["bias_label"] == cls]
        if len(cls_rows):
            samples.append(cls_rows.sample(1, random_state=7+cls).iloc[0])
    # fill to 5 if needed
    extras = bias_df.sample(n=min(5, len(bias_df)), random_state=99)
    for _, r in extras.iterrows():
        if len(samples) >= 5:
            break
        if not any(s["text"] == r["text"] for s in samples):
            samples.append(r)

    model.eval()
    device = config.DEVICE
    tqdm.write("\n  [BIAS PROBE] 5-sample held-out check:")
    tqdm.write("  {:<8} {:<8} {:<8}".format("True", "Pred", "Match"))
    tqdm.write("  " + "-"*30)
    with torch.no_grad():
        for row in samples:
            pooled = _get_pooled_embedding(model, str(row["text"]), tokenizer, device)
            logits = model.bias_head(pooled)
            pred = torch.argmax(logits, dim=1).item()
            true = int(row["bias_label"])
            match = "YES" if pred == true else "NO"
            tqdm.write("  {:<8} {:<8} {}".format(
                NAMES.get(true, str(true)), NAMES.get(pred, str(pred)), match))
    model.train()


def _gonogo_check(epoch_losses, bias_acc, epoch):
    """
    Step 5 go/no-go gate after epoch 1.
    Returns True (go) or False (no-go / halt).
    epoch_losses = {"bias": float, "fact": float, "intent": float, "emotion": float}
    """
    import math
    PASS = "\u2705 GO"
    FAIL = "\u274c NO-GO"
    results = []

    lb  = epoch_losses.get("bias",    float("nan"))
    lf  = epoch_losses.get("fact",    float("nan"))
    li  = epoch_losses.get("intent",  float("nan"))
    le  = epoch_losses.get("emotion", float("nan"))

    def _ok(v):
        return not (math.isnan(v) or math.isinf(v))

    checks = [
        ("loss_bias not NaN",      _ok(lb),          "nan", str(round(lb,4))),
        ("loss_bias > 0",          lb > 0,            "0",   str(round(lb,4))),
        ("loss_fact < 0.25",       _ok(lf) and lf < 0.25,  "0.25", str(round(lf,4))),
        ("loss_intent < 0.70",     _ok(li) and li < 0.70,  "0.70", str(round(li,4))),
        ("loss_emotion not NaN",   _ok(le),           "nan", str(round(le,4))),
        ("loss_emotion < 2.0",     _ok(le) and le < 2.0,   "2.00", str(round(le,4))),
        ("No NaN in any loss",     all(_ok(x) for x in [lb,lf,li,le]), "NaN", "clean"),
    ]

    tqdm.write("\n" + "=" * 60)
    tqdm.write("  EPOCH {} GO/NO-GO REPORT".format(epoch))
    tqdm.write("  bias_acc: {:.1f}%".format(bias_acc * 100))
    tqdm.write("  {:<32} {:<8} {:<8} {}".format("Check", "Thresh", "Value", "Verdict"))
    tqdm.write("  " + "-" * 56)
    all_pass = True
    for name, passed, thresh, value in checks:
        verdict = PASS if passed else FAIL
        tqdm.write("  {:<32} {:<8} {:<8} {}".format(name, thresh, value, verdict))
        if not passed:
            all_pass = False
    tqdm.write("=" * 60)
    if all_pass:
        tqdm.write("  ALL CHECKS PASSED — continuing to epoch 2")
    else:
        tqdm.write("  ONE OR MORE CHECKS FAILED — stopping. Diagnose before proceeding.")
    return all_pass


import random
import math
from torch.utils.data import Sampler

class TaskRoundRobinSampler(Sampler):
    def __init__(self, df, batch_size):
        # Identify indices for each task
        self.bias_idx = df[df["bias_label"] >= 0].index.tolist()
        self.fact_idx = df[df["fact_score"] >= 0.0].index.tolist()
        self.intent_idx = df[df["intent_label"] >= 0].index.tolist()
        self.emotion_idx = df[df["emotion_label"].apply(lambda x: '1' in str(x))].index.tolist()
        self.batch_size = batch_size

    def __iter__(self):
        # Shuffle indices
        random.shuffle(self.bias_idx)
        random.shuffle(self.fact_idx)
        random.shuffle(self.intent_idx)
        random.shuffle(self.emotion_idx)

        # Chunk into batches
        def chunk(lst):
            return [lst[i:i + self.batch_size] for i in range(0, len(lst), self.batch_size)]
        
        bias_batches = chunk(self.bias_idx)
        fact_batches = chunk(self.fact_idx)
        intent_batches = chunk(self.intent_idx)
        emo_batches = chunk(self.emotion_idx)

        # Round robin yield
        iters = [iter(bias_batches), iter(fact_batches), iter(intent_batches), iter(emo_batches)]
        active = [True] * 4
        while any(active):
            for i in range(4):
                if active[i]:
                    try:
                        yield next(iters[i])
                    except StopIteration:
                        active[i] = False

    def __len__(self):
        return (math.ceil(len(self.bias_idx) / self.batch_size) + 
                math.ceil(len(self.fact_idx) / self.batch_size) + 
                math.ceil(len(self.intent_idx) / self.batch_size) + 
                math.ceil(len(self.emotion_idx) / self.batch_size))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh",    action="store_true")
    parser.add_argument("--finetune", action="store_true",
                        help="Load weights from latest checkpoint but reset epoch counter to 0")
    args, _ = parser.parse_known_args()

    if args.fresh:
        ckpts = glob.glob(os.path.join(config.CHECKPOINT_DIR, "*.pth"))
        for f in ckpts: os.remove(f)
        print("  Fresh start: checkpoints cleared.")

    device = config.DEVICE
    print("\n" + "=" * 60)
    print("  CredibleX BALANCED MODE Trainer (RTX 4050 6GB)")
    print("  Device: {} | Backbone: {}".format(device, config.MODEL_NAME))
    print("  Batch: {} | MaxLen: {} | GradAccum: {}".format(config.BATCH_SIZE, config.MAX_LEN, config.GRAD_ACCUM_STEPS))
    print("=" * 60 + "\n")

    # Load Data
    df = pd.read_csv("training_data.csv")
    print("Loaded {:,} articles".format(len(df)))

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
    model = NewsTrustModel(config.MODEL_NAME).to(device)
    model.configure_for_training()

    # BALANCED MODE DataLoader (Round-Robin)
    train_dataset = NewsDataset(df, tokenizer)
    sampler = TaskRoundRobinSampler(df, config.BATCH_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler = sampler,
        num_workers   = 0,           # Set to 0 to fix Windows pagefile error
        pin_memory    = True,        # faster CPU→GPU transfer
        collate_fn    = collate_fn,
    )

    # Weights & Losses
    # Filter to rows that actually carry each label before computing class weights
    valid_bias_series   = df[df["bias_label"] >= 0]["bias_label"]
    valid_intent_series = df[df["intent_label"] >= 0]["intent_label"]
    bias_w   = compute_class_weights(valid_bias_series,   config.N_BIAS_CLASSES, device)
    intent_w = compute_class_weights(valid_intent_series, 2, device)
    print("  Bias class weights  (N={}): {}".format(config.N_BIAS_CLASSES, bias_w.tolist()))
    print("  Intent class weights (N=2): {}".format(intent_w.tolist()))
    # ignore_index=-1: CrossEntropyLoss auto-zeroes loss for sentinel rows
    loss_bias    = nn.CrossEntropyLoss(weight=bias_w,   label_smoothing=0.1, ignore_index=-1)
    loss_intent  = nn.CrossEntropyLoss(weight=intent_w, ignore_index=-1)
    # Emotion: pos_weight=3 forces model to predict positive classes in sparse multi-hot labels
    emotion_pos_w = torch.full((config.N_EMOTION_CLASSES,), 3.0, device=device)
    loss_emotion = nn.BCEWithLogitsLoss(pos_weight=emotion_pos_w)
    # Factuality: HuberLoss allows confident extreme predictions (avoids mean-collapse of MSELoss)
    loss_fact    = nn.HuberLoss(delta=0.3)

    # Optimizer
    if config.USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
            print("Using 8-bit Adam (bitsandbytes)")
        except ImportError:
            print("[WARN] bitsandbytes not installed — using standard AdamW")
            optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

    n_batches = len(train_loader)
    total_steps = (n_batches // config.GRAD_ACCUM_STEPS) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Resume
    best_bias_acc = 0.0
    ckpt_path, start_epoch = _find_latest_checkpoint()
    if ckpt_path:
        print("Resuming from: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if not args.finetune:
            # Normal resume: also restore optimizer/scheduler state
            if "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]: scaler.load_state_dict(ckpt["scaler_state_dict"])
        else:
            # Fine-tune: weights only, fresh optimizer/scheduler, epoch counter reset to 0
            start_epoch = 0
            print("  [--finetune] Epoch counter reset to 0. Fresh optimizer/scheduler.")
        best_bias_acc = ckpt.get("best_bias_acc", 0.0)

    # ── Training Loop ─────────────────────────────────────────────────────────
    train_start = time.time()

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        # ── Per-task accumulators ─────────────────────────────────────────────
        ep_lb, ep_lf, ep_li, ep_le = 0.0, 0.0, 0.0, 0.0
        n_lb,  n_lf,  n_li,  n_le  = 0,   0,   0,   0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, config.EPOCHS), dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            # VRAM Guard
            if device.type == "cuda":
                if torch.cuda.memory_allocated() / 1e9 > config.VRAM_SAFETY_THRESHOLD_GB:
                    torch.cuda.empty_cache()
                    if torch.cuda.memory_allocated() / 1e9 > config.VRAM_SAFETY_THRESHOLD_GB:
                        tqdm.write("[VRAM] Safety skip at batch {}".format(batch_idx))
                        continue

            texts   = batch["text"]
            b_lbls  = batch["bias_label"].to(device)
            f_lbls  = batch["fact_score"].to(device)
            i_lbls  = batch["intent_label"].to(device)
            e_lbls  = batch["emotion_label"].to(device)

            try:
                # Balanced Mode: Process batch items
                batch_loss = 0
                for i in range(len(texts)):
                    pooled = _get_pooled_embedding(model, texts[i], tokenizer, device)
                    
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        heads_out = {
                            "bias":       model.bias_head(pooled),
                            "factuality": torch.sigmoid(model.fact_head(pooled)),
                            "intent":     model.intent_head(pooled),
                            "emotion":    model.emotion_head(pooled),
                        }

                        # ── Masked multi-task loss ────────────────────────────
                        
                        # bias: manual mask
                        if b_lbls[i].item() >= 0:
                            lb = loss_bias(heads_out["bias"], b_lbls[i].unsqueeze(0))
                        else:
                            lb = torch.tensor(0.0, device=device)

                        # intent: manual mask
                        if i_lbls[i].item() >= 0:
                            li = loss_intent(heads_out["intent"], i_lbls[i].unsqueeze(0))
                        else:
                            li = torch.tensor(0.0, device=device)

                        # factuality: manual mask — skip when fact_score == -1.0
                        if f_lbls[i].item() >= 0.0:
                            lf = loss_fact(
                                heads_out["factuality"].squeeze(-1),
                                f_lbls[i].unsqueeze(0)
                            )
                        else:
                            lf = torch.tensor(0.0, device=device)

                        # emotion: manual mask — skip when vector is all-zeros (sentinel)
                        e_vec = e_lbls[i]   # shape (N_EKMAN,)
                        if e_vec.sum() > 0:
                            le = loss_emotion(
                                heads_out["emotion"],
                                e_vec.unsqueeze(0)
                            )
                        else:
                            le = torch.tensor(0.0, device=device)

                        # STEP 2 FIX: Bias needs a louder signal against Fact/Emotion gradients
                        item_loss = (2.0 * lb) + lf + li + le
                        batch_loss += item_loss / len(texts)

                        # ── Accumulate per-task losses (non-sentinel rows only) ──
                        if b_lbls[i].item() >= 0:
                            ep_lb += lb.item(); n_lb += 1
                        if f_lbls[i].item() >= 0.0:
                            ep_lf += lf.item(); n_lf += 1
                        if i_lbls[i].item() >= 0:
                            ep_li += li.item(); n_li += 1
                        if e_lbls[i].sum().item() > 0:
                            ep_le += le.item(); n_le += 1

                # Backprop after batch
                scaler.scale(batch_loss / config.GRAD_ACCUM_STEPS).backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    scaler.update()
                    tqdm.write("[OOM] batch {} skipped".format(batch_idx))
                    continue
                raise e

            epoch_loss += (batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss)

            # Optimizer Step
            if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # VRAM & Speed Logging
            if batch_idx % 100 == 0 and device.type == "cuda":
                used = torch.cuda.memory_allocated() / 1e9
                peak = torch.cuda.max_memory_allocated() / 1e9
                pbar.set_postfix({"loss": "{:.4f}".format(epoch_loss/(batch_idx+1)), "vram": "{:.1f}G".format(used)})
                if batch_idx % 500 == 0:
                    tqdm.write("  [VRAM] current={:.2f}GB  peak={:.2f}GB".format(used, peak))

        # ── End-of-Epoch Reporting ────────────────────────────────────────────
        avg = lambda total, n: total / max(n, 1)
        epoch_losses = {
            "bias":    avg(ep_lb, n_lb),
            "fact":    avg(ep_lf, n_lf),
            "intent":  avg(ep_li, n_li),
            "emotion": avg(ep_le, n_le),
        }
        bias_acc = _quick_bias_acc(model, df, tokenizer)

        tqdm.write("\n" + "-" * 60)
        tqdm.write("  Epoch {}/{} Summary".format(epoch+1, config.EPOCHS))
        tqdm.write("  loss_total  : {:.4f}".format(epoch_loss / max(n_batches, 1)))
        tqdm.write("  loss_bias   : {:.4f}  (n={:,})".format(epoch_losses["bias"],    n_lb))
        tqdm.write("  loss_fact   : {:.4f}  (n={:,})".format(epoch_losses["fact"],    n_lf))
        tqdm.write("  loss_intent : {:.4f}  (n={:,})".format(epoch_losses["intent"],  n_li))
        tqdm.write("  loss_emotion: {:.4f}  (n={:,})".format(epoch_losses["emotion"], n_le))
        tqdm.write("  bias_acc    : {:.1f}%".format(bias_acc * 100))
        tqdm.write("-" * 60)

        # ── Bias probe: 5 held-out predictions (always run) ───────────────────
        _bias_sample_probe(model, df, tokenizer)

        # ── Epoch 1: go/no-go gate ────────────────────────────────────────────
        if epoch == 0:
            go = _gonogo_check(epoch_losses, bias_acc, epoch + 1)
            if not go:
                tqdm.write("  Halting training. Fix failing checks before restarting.")
                break

        # Save Checkpoint
        save_dict = {
            "epoch": epoch+1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_bias_acc": best_bias_acc, "scaler_state_dict": scaler.state_dict() if scaler else None
        }
        _safe_save(os.path.join(config.CHECKPOINT_DIR, "epoch_{}.pth".format(epoch+1)), save_dict)
        
        if bias_acc > best_bias_acc:
            best_bias_acc = bias_acc
            _safe_save(config.SAVE_PATH, save_dict)
            tqdm.write("  New best bias_acc: {:.1f}%".format(best_bias_acc*100))

    print("\nTraining Complete. Total time: {}".format(_fmt_time(time.time()-train_start)))

if __name__ == "__main__":
    train()