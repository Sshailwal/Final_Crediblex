import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import argparse

import config
from model import NewsTrustModel
from train import NewsDataset, collate_fn, compute_class_weights, _safe_save

def quick_bias_acc(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            b_lbls = batch["bias_label"].to(device)
            mask = (b_lbls >= 0)
            if not mask.any(): continue
            
            texts = [batch["text"][i] for i in range(len(batch["text"])) if mask[i]]
            b_lbls = b_lbls[mask]
            
            # Simple encoding since this is just validation
            window_embeddings = []
            from train import _encode_windows
            for i in range(len(texts)):
                windows = []
                for input_ids, att_mask in _encode_windows(texts[i], val_loader.dataset.tokenizer, device, config.MAX_LEN, config.SLIDING_STRIDE):
                    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                        out = model.backbone(input_ids=input_ids, attention_mask=att_mask)
                        p = model._mean_pool(out.last_hidden_state, att_mask)
                    windows.append(p)
                pooled = torch.stack(windows).mean(dim=0)
                window_embeddings.append(pooled)
            
            pooled_batch = torch.cat(window_embeddings, dim=0)
            logits = model.bias_head(pooled_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_lbls).sum().item()
            total += len(b_lbls)
            
    return correct / total if total > 0 else 0.0

def train():
    device = config.DEVICE
    print(f"--- BIAS-ONLY FINE-TUNE (Augmented Data) ---")
    
    # Load augmented data (only bias rows)
    df = pd.read_csv("training_data_augmented.csv")
    bias_df = df[df["bias_label"] >= 0].copy()
    print(f"Loaded {len(bias_df)} bias rows for fine-tuning")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
    ds = NewsDataset(bias_df, tokenizer)
    train_loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Load model and checkpoint
    model = NewsTrustModel(config.MODEL_NAME).to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "epoch_3.pth")
    print(f"Resuming from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # FREEZING LOGIC
    # 1. Freeze ALL heads except bias_head
    for param in model.fact_head.parameters(): param.requires_grad = False
    for param in model.intent_head.parameters(): param.requires_grad = False
    for param in model.emotion_head.parameters(): param.requires_grad = False
    
    # 2. Freeze backbone layers 0-8, leave 9-11 + pooler trainable
    # DeBERTa layers are in model.backbone.encoder.layer
    for i in range(9):
        for param in model.backbone.encoder.layer[i].parameters():
            param.requires_grad = False
    for param in model.backbone.embeddings.parameters():
        param.requires_grad = False
        
    print("Frozen heads: Factuality, Intent, Emotion")
    print("Frozen backbone layers: 0-8 and embeddings")
    
    # Check class weights
    bias_w = compute_class_weights(bias_df["bias_label"], config.N_BIAS_CLASSES, device)
    print(f"Bias class weights: {bias_w.tolist()}")
    loss_bias = nn.CrossEntropyLoss(weight=bias_w, label_smoothing=0.1, ignore_index=-1)
    
    # Optimizer & Scheduler (LR: 1e-5)
    from torch.optim import AdamW
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.01)
    
    epochs = 5
    n_batches = len(train_loader)
    total_steps = (n_batches // config.GRAD_ACCUM_STEPS) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Bias Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            texts = batch["text"]
            b_lbls = batch["bias_label"].to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                # manual forward
                window_embeddings = []
                from train import _encode_windows
                for i in range(len(texts)):
                    windows = []
                    for input_ids, att_mask in _encode_windows(texts[i], tokenizer, device, config.MAX_LEN, config.SLIDING_STRIDE):
                        out = model.backbone(input_ids=input_ids, attention_mask=att_mask)
                        p = model._mean_pool(out.last_hidden_state, att_mask)
                        p = model.dropout(p)
                        windows.append(p)
                    pooled = torch.stack(windows).mean(dim=0)
                    window_embeddings.append(pooled)
                
                pooled_batch = torch.cat(window_embeddings, dim=0)
                bias_logits = model.bias_head(pooled_batch)
                loss = loss_bias(bias_logits, b_lbls)
                
            if scaler:
                scaler.scale(loss / config.GRAD_ACCUM_STEPS).backward()
            else:
                (loss / config.GRAD_ACCUM_STEPS).backward()
                
            epoch_loss += loss.item() * len(texts)
            n_samples += len(texts)
            
            if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            pbar.set_postfix({"loss": f"{epoch_loss/n_samples:.4f}"})
            
        avg_loss = epoch_loss / n_samples
        
        # Validation accuracy on original (non-augmented) rows
        val_df = bias_df[bias_df["_source"] != "babe_augmented"].sample(200, random_state=epoch)
        val_ds = NewsDataset(val_df, tokenizer)
        val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)
        acc = quick_bias_acc(model, val_loader, device)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"loss_bias: {avg_loss:.4f}")
        print(f"bias_acc:  {acc*100:.1f}%")
        
        save_dict = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict()
        }
        _safe_save(os.path.join(config.CHECKPOINT_DIR, f"bias_ft_epoch_{epoch+1}.pth"), save_dict)
        
        if acc > best_acc:
            best_acc = acc
            _safe_save(os.path.join(config.CHECKPOINT_DIR, "best_bias_finetuned.pth"), save_dict)
            
        if acc > 0.65:
            print(f"\n🎯 Target >65% achieved ({acc*100:.1f}%)! Stopping early.")
            break

if __name__ == "__main__":
    train()
