# emotion_label: int→multihot
import pandas as pd
from datasets import load_dataset
from schema import ArticleRecord
from tqdm import tqdm
import os
import sys
import config
import json

def to_multihot(labels, n=28):
    """Convert single label or list of labels to JSON multi-hot string."""
    if isinstance(labels, (int, float)):
        labels = [int(labels)]
    vec = [0.0] * n
    for l in labels:
        if 0 <= l < n:
            vec[l] = 1.0
    return json.dumps(vec)

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Force UTF-8 output on Windows so print statements don't crash on emoji
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# Emotion -> factuality proxy  (GoEmotions 28-class)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_FACT_SCORE = {
    0: 0.80, 1: 0.70, 2: 0.20, 3: 0.30,
    4: 0.85, 5: 0.75, 6: 0.50, 7: 0.65,
    8: 0.60, 9: 0.30, 10: 0.20, 11: 0.10,
    12: 0.35, 13: 0.70, 14: 0.20, 15: 0.90,
    16: 0.25, 17: 0.80, 18: 0.80, 19: 0.30,
    20: 0.70, 21: 0.70, 22: 0.60, 23: 0.75,
    24: 0.30, 25: 0.30, 26: 0.55, 27: 0.65,
}


def upsample_minority_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample minority bias rows to reduce class imbalance."""
    class_dfs = {}
    for c in range(config.N_BIAS_CLASSES):
        class_dfs[c] = df[df['bias_label'] == c].copy()

    before_str = "  ".join(f"Class {c}: {len(class_dfs[c]):,}" for c in range(config.N_BIAS_CLASSES))
    print(f"\n[BIAS] Before upsample -> {before_str}")

    max_class_count = max(len(cdf) for cdf in class_dfs.values())
    target = min(max_class_count // 2, 8_000)

    def _up(sub, t):
        if len(sub) == 0 or len(sub) >= t:
            return sub
        r = (t // len(sub)) + 1
        return pd.concat([sub] * r, ignore_index=True).head(t)

    balanced_dfs = []
    for c in range(config.N_BIAS_CLASSES):
        balanced_dfs.append(_up(class_dfs[c], target))

    balanced = pd.concat(balanced_dfs, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    lc = balanced['bias_label'].value_counts().to_dict()
    after_str = "  ".join(f"Class {c}: {lc.get(c, 0):,}" for c in range(config.N_BIAS_CLASSES))
    print(f"[BIAS] After  upsample -> {after_str}")
    print(f"Total rows: {len(balanced):,}\n")
    return balanced


def get_explainable_dataset() -> pd.DataFrame:
    print("=" * 70)
    print("  Downloading & assembling CredibleX training dataset")
    print("=" * 70)

    # Bug 11: Migration Note — emotion_label is now a JSON multi-hot string.
    # Existing training_data.csv MUST be regenerated to avoid shape mismatch.

    unified_data = []
    liar_loaded  = False

    # SOURCE 1: LIAR -- factuality labels
    try:
        print("\n[1/5] LIAR dataset (factuality signal)...")
        liar_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
        liar = load_dataset("UKPLab/liar")
        before = len(unified_data)
        for row in tqdm(liar['train'], desc="  LIAR"):
            try:
                unified_data.append(ArticleRecord(
                    text=str(row['text']), bias_label=1,
                    fact_score=liar_map.get(int(row['labels']), 0.5),
                    intent_label=0, emotion_label=to_multihot(27),
                ).model_dump())
            except Exception:
                continue
        liar_loaded = True
        print(f"  [OK] LIAR: +{len(unified_data) - before:,} rows")
    except Exception as e:
        print(f"  [SKIP] LIAR unavailable: {e}")

    # SOURCE 2: MBIB -- real Left / Center / Right labels
    try:
        print("\n[2/5] MBIB (political bias -- real Left/Center/Right)...")
        mbib = load_dataset("mediabiasgroup/mbib-base", split="political_bias")
        count = 0
        for row in tqdm(mbib, desc="  MBIB"):
            if count >= 5_000:
                break
            try:
                raw_lbl  = int(row['label'])
                bias_lbl = raw_lbl if raw_lbl in (0, 1, 2) else 1
                unified_data.append(ArticleRecord(
                    text=str(row['text']), bias_label=bias_lbl,
                    fact_score=0.5, intent_label=0, emotion_label=to_multihot(27),
                ).model_dump())
                count += 1
            except Exception:
                continue
        print(f"  [OK] MBIB: +{count:,} rows (real Left/Center/Right)")
    except Exception as e:
        print(f"  [SKIP] MBIB unavailable: {e}")

    # SOURCE 3: Hyperpartisan News -- Right-leaning signal
    try:
        print("\n[3/5] Hyperpartisan News (Right-leaning signal)...")
        hyper = load_dataset("zapsdcn/hyperpartisan_news", split="train")
        count = 0
        for row in tqdm(hyper, desc="  Hyperpartisan"):
            if count >= 4_000:
                break
            try:
                is_hyper = str(row.get('label', '')).lower() == 'true'
                unified_data.append(ArticleRecord(
                    text=str(row['text']),
                    bias_label=2 if is_hyper else 1,
                    fact_score=0.3 if is_hyper else 0.7,
                    intent_label=1 if is_hyper else 0,
                    emotion_label=to_multihot(27),
                ).model_dump())
                count += 1
            except Exception:
                continue
        print(f"  [OK] Hyperpartisan: +{count:,} rows")
    except Exception as e:
        print(f"  [SKIP] Hyperpartisan unavailable: {e}")

    # SOURCE 4: Fake News / Satire
    try:
        print("\n[4/5] Fake News dataset (satire / intent signal)...")
        fake = load_dataset("GonzaloA/fake_news", split="train")
        count = 0
        for row in tqdm(fake, desc="  FakeNews"):
            if count >= 3_000:
                break
            try:
                is_fake = int(row['label']) == 0
                unified_data.append(ArticleRecord(
                    text=str(row['text']), bias_label=1,
                    fact_score=0.1 if is_fake else 0.8,
                    intent_label=2 if is_fake else 0,
                    emotion_label=to_multihot(27),
                ).model_dump())
                count += 1
            except Exception:
                continue
        print(f"  [OK] Fake News: +{count:,} rows")
    except Exception as e:
        print(f"  [SKIP] Fake News unavailable: {e}")

    # SOURCE 5: GoEmotions -- emotion labels
    print("\n[5/5] GoEmotions (emotion signal)...")
    TARGET_EMOTIONS = 5_000 if liar_loaded else 10_000
    try:
        emotions      = load_dataset("google-research-datasets/go_emotions", "simplified")
        emotion_train = emotions['train']
        count = 0
        for row in tqdm(emotion_train, desc="  GoEmotions"):
            if count >= TARGET_EMOTIONS:
                break
            try:
                text = str(row.get('text', ''))
                if not text or len(text) < 10:
                    continue
                labels = row.get('labels', [])
                emo    = labels[0] if isinstance(labels, list) and labels else 27
                unified_data.append(ArticleRecord(
                    text=text, bias_label=1,
                    fact_score=EMOTION_FACT_SCORE.get(int(emo), 0.5),
                    intent_label=0, emotion_label=to_multihot(labels),
                ).model_dump())
                count += 1
            except Exception:
                continue
        print(f"  [OK] GoEmotions: +{count:,} rows")
    except Exception as e:
        print(f"  [SKIP] GoEmotions unavailable: {e}")

    print(f"\n{'='*70}")
    print(f"  Raw total: {len(unified_data):,} rows assembled from all sources")
    print(f"{'='*70}")

    if not unified_data:
        raise RuntimeError("No data collected -- all sources failed. Check internet connectivity.")

    df = pd.DataFrame(unified_data)
    before = len(df)
    df = df.dropna(subset=['text', 'bias_label', 'fact_score', 'intent_label', 'emotion_label'])
    df = df[df['text'].str.strip().str.len() >= 10]
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} invalid/short rows")

    df = upsample_minority_bias(df)
    return df


if __name__ == "__main__":
    df = get_explainable_dataset()
    if len(df) > 0:
        df.to_csv("training_data.csv", index=False)
        print(f"[SAVED] training_data.csv  ({len(df):,} rows)")
        print("\nFinal bias distribution:")
        vc = df['bias_label'].value_counts()
        for c in range(config.N_BIAS_CLASSES):
            print(f"  Class {c}: {vc.get(c, 0):,}")
        print("\nFinal intent distribution:")
        vi = df['intent_label'].value_counts()
        print(f"  News(0)   : {vi.get(0, 0):,}")
        print(f"  Opinion(1): {vi.get(1, 0):,}")
        print(f"  Satire(2) : {vi.get(2, 0):,}")
        print(f"\nfact_score range: {df['fact_score'].min():.2f} - {df['fact_score'].max():.2f}")
    else:
        print("[ERROR] No data collected.")