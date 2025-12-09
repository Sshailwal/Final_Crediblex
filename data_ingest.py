import pandas as pd
from datasets import load_dataset
from schema import ArticleRecord
from tqdm import tqdm

def get_unified_dataset():
    print("⬇️  Downloading Datasets via HuggingFace...")
    
    # 1. Load LIAR (Factuality Data)
    # LIAR labels: 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
    try:
        liar = load_dataset("liar")
    except Exception as e:
        print(f"❌ Error loading LIAR dataset: {e}")
        liar = None
    
    # 2. Load GoEmotions (Emotion Data)
    try:
        emotions = load_dataset("go_emotions", "simplified")
    except Exception as e:
        print(f"❌ Error loading GoEmotions dataset: {e}")
        emotions = None
    
    unified_data = []
    
    # Process LIAR Data
    if liar is not None:
        print("🔄 Processing LIAR Data (Factuality & Bias)...")
        # We map LIAR's 6 labels to a 0.0 - 1.0 score
        liar_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
        
        for row in tqdm(liar['train']):
            try:
                # Check if statement exists and is not empty
                if not row.get('statement') or not isinstance(row['statement'], str):
                    continue
                    
                record = ArticleRecord(
                    text=row['statement'],
                    bias_label=1,  # LIAR doesn't have bias labels, so we set 'Center' (1) as placeholder
                    fact_score=liar_map.get(row['label'], 0.5),
                    intent_label=0,  # Assume News
                    emotion_label=0  # Assume Neutral placeholder
                )
                unified_data.append(record.model_dump())
            except Exception as e:
                # Silently skip problematic rows
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
    df = get_unified_dataset()
    
    if len(df) > 0:
        df.to_csv("training_data.csv", index=False)
        print("💾 Saved to training_data.csv")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
    else:
        print("⚠️  No data was collected. Please check dataset availability.")