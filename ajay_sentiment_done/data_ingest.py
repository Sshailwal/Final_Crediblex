import pandas as pd
from datasets import load_dataset
from schema import ArticleRecord
from tqdm import tqdm
import os

# Suppress HuggingFace cache warnings for paths with spaces
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def get_explainable_dataset():
    print("⬇️ Downloading Explainable Datasets via HuggingFace...")
    
    # LOAD LIAR-PLUS (Contains Justifications)
    try:
        liar_plus = load_dataset("ucsbnlp/liar")
    except Exception as e:
        print(f"Failed to load LIAR dataset: {e}")
        liar_plus = None
    
    # LOAD GO-EMOTIONS
    emotions = load_dataset("go_emotions", "simplified")

    unified_data = []

    print("🔄 Processing LIAR-PLUS (Extracting Explanations)...")
    liar_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
    
    if liar_plus:
        for row in tqdm(liar_plus['train']):
            try:
                record = ArticleRecord(
                    text=row['statement'],
                    bias_label=1, 
                    fact_score=liar_map.get(row['label'], 0.5),
                    intent_label=0, 
                    emotion_label=0 
                )
                unified_data.append(record.dict())
            except:
                continue
            
    
    # Process GoEmotions Data
    if emotions is not None:
        print("🔄 Processing GoEmotions Data (Emotion)...")
        # We take a slice so training doesn't take forever
        emotion_train = emotions['train']
        
        sample_count = 0
        for row in tqdm(emotion_train):
            if sample_count >= 5000:
                break
            try:
                # Extract text from row
                text = row['text']
                if not text or not isinstance(text, str):
                    continue
                
                # Taking the first emotion found, default to 0 if none
                labels = row['labels']
                emo = labels[0] if isinstance(labels, list) and len(labels) > 0 else 0
                
                record = ArticleRecord(
                    text=text,
                    bias_label=1,  # Placeholder
                    fact_score=0.5,  # Unknown factuality
                    intent_label=1,  # Reddit comments are usually 'Opinion'
                    emotion_label=emo  # Real emotion label
                )
                unified_data.append(record.model_dump())
                sample_count += 1
            except Exception as e:
                # Silently skip problematic rows
                continue
    
    print(f"✅ Created Unified Dataset with {len(unified_data)} samples.")
    return pd.DataFrame(unified_data)

if __name__ == "__main__":
    df = get_explainable_dataset()
    
    if len(df) > 0:
        df.to_csv("training_data.csv", index=False)
        print("💾 Saved to training_data.csv")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
    else:
        print("⚠️  No data was collected. Please check dataset availability.")