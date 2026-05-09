import pandas as pd
import torch
from transformers import pipeline
import time

print("Loading data...")
df = pd.read_csv("training_data.csv")
bias_df = df[df["bias_label"] >= 0].copy()

print(f"Total bias rows to augment: {len(bias_df)}")

device = 0 if torch.cuda.is_available() else -1
print(f"Loading translation models on device {device}...")
translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr", device=device)
translator_fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", device=device)

start = time.time()
augmented_rows = []

from tqdm import tqdm
batch_size = 16

texts = bias_df["text"].tolist()
print(f"Translating EN -> FR (batch_size={batch_size})...")
fr_texts = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    out = translator_en_fr(batch, max_length=512)
    fr_texts.extend([x["translation_text"] for x in out])

print(f"Translating FR -> EN (batch_size={batch_size})...")
en_texts = []
for i in tqdm(range(0, len(fr_texts), batch_size)):
    batch = fr_texts[i:i+batch_size]
    out = translator_fr_en(batch, max_length=512)
    en_texts.extend([x["translation_text"] for x in out])

for i, (_, row) in enumerate(bias_df.iterrows()):
    new_row = row.copy()
    new_row["text"] = en_texts[i]
    new_row["_source"] = "babe_augmented"
    augmented_rows.append(new_row)

aug_df = pd.DataFrame(augmented_rows)
print(f"\nOriginal rows: {len(df)}")
final_df = pd.concat([df, aug_df], ignore_index=True)
print(f"Final rows: {len(final_df)}")

final_df.to_csv("training_data_augmented.csv", index=False)
print(f"Saved to training_data_augmented.csv. Total time: {time.time()-start:.1f}s")
