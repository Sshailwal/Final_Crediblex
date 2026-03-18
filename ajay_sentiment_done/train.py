import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
import pandas as pd
import os
from functools import partial
from data_ingest import get_explainable_dataset
from model import NewsTrustModel
import config

# Suppress HuggingFace cache warnings for paths with spaces
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Global tokenizer used by collate_fn (top-level so it's picklable for Windows spawn)
TOKENIZER = None

# 1. Prepare Data Loader
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Return raw row values; batch tokenization is handled in collate_fn
        row = self.df.iloc[idx]
        return {
            'text': row['text'],
            'bias_label': torch.tensor(row['bias_label'], dtype=torch.long),
            'fact_label': torch.tensor(row['fact_score'], dtype=torch.float),
            'intent_label': torch.tensor(row['intent_label'], dtype=torch.long),
            'emotion_label': torch.tensor(row['emotion_label'], dtype=torch.long)
        }


def collate_fn(batch, max_len=512):
    # batch: list of dicts returned by __getitem__
    # Use the module-level TOKENIZER so this function is picklable on Windows
    if TOKENIZER is None:
        raise RuntimeError('TOKENIZER is not initialized. Set TOKENIZER before creating DataLoader.')

    texts = [item['text'] for item in batch]
    encoding = TOKENIZER(
        texts,
        max_length=max_len,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'bias_label': torch.stack([item['bias_label'] for item in batch]).long(),
        'fact_label': torch.stack([item['fact_label'] for item in batch]).float(),
        'intent_label': torch.stack([item['intent_label'] for item in batch]).long(),
        'emotion_label': torch.stack([item['emotion_label'] for item in batch]).long()
    }

# 2. Training Loop
def train():
    # Load Data (If csv exists use it, else generate)
    try:
        df = pd.read_csv("training_data.csv")
        print("📂 Loaded data from CSV.")
    except:
        df = get_explainable_dataset()
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
    # set global tokenizer for collate_fn (needed for multiprocessing on Windows)
    global TOKENIZER
    TOKENIZER = tokenizer
    dataset = NewsDataset(df, tokenizer, config.MAX_LEN)
    # Use batch tokenization via collate_fn, increase workers and pin_memory for GPU
    # Note: On Windows, num_workers > 0 may cause multiprocessing issues; use 0 for single-process loading
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(config.DEVICE == 'cuda'),
        collate_fn=partial(collate_fn, max_len=config.MAX_LEN)
    )
    
    # Load Model
    model = NewsTrustModel(config.MODEL_NAME).to(config.DEVICE)
    
    # Enable GPU optimizations
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        # We'll use torch.cuda.amp for mixed precision (safer than forcing .half())
        print(f"🎯 GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Loss Functions
    loss_fn_class = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss() # For factuality score (regression)
    
    # Initialize scaler for GPU (won't be used on CPU but declared for clarity)
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE == 'cuda' else None
    
    print("🚀 Starting Training...")
    model.train()
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # Move batch to GPU/CPU
            input_ids = batch['input_ids'].to(config.DEVICE)
            mask = batch['attention_mask'].to(config.DEVICE)
            bias_labels = batch['bias_label'].to(config.DEVICE)
            fact_labels = batch['fact_label'].to(config.DEVICE)
            intent_labels = batch['intent_label'].to(config.DEVICE)
            emotion_labels = batch['emotion_label'].to(config.DEVICE)

            # Forward + backward with AMP when on GPU
            if config.DEVICE == 'cuda':
                assert scaler is not None, "scaler should not be None when using CUDA"
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, mask)
                    loss_bias = loss_fn_class(outputs['bias'], bias_labels)
                    loss_fact = loss_fn_reg(outputs['factuality'].squeeze(), fact_labels)
                    loss_intent = loss_fn_class(outputs['intent'], intent_labels)
                    loss_emotion = loss_fn_class(outputs['emotion'], emotion_labels)
                    loss = loss_bias + loss_fact + loss_intent + loss_emotion

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, mask)
                loss_bias = loss_fn_class(outputs['bias'], bias_labels)
                loss_fact = loss_fn_reg(outputs['factuality'].squeeze(), fact_labels)
                loss_intent = loss_fn_class(outputs['intent'], intent_labels)
                loss_emotion = loss_fn_class(outputs['emotion'], emotion_labels)
                loss = loss_bias + loss_fact + loss_intent + loss_emotion

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss/len(loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {avg_loss:.4f}", end="")
        if config.DEVICE == "cuda":
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(f" | GPU Memory: {gpu_mem:.2f} GB")
        else:
            print()

    # Save Model
    torch.save(model.state_dict(), config.SAVE_PATH)
    print(f"✅ Model saved to {config.SAVE_PATH}")

if __name__ == "__main__":
    train()