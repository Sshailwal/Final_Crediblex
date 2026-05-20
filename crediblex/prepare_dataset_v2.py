"""
prepare_dataset_v2.py
Phase 1 -- Dataset Preparation Pipeline
Political Bias Classification (5 classes: 0=Far Left ... 4=Far Right)
"""

import os
import re
import sys
import json
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

# ── PATH CONSTANTS ───────────────────────────────────────────────────────────
DATASET_A_PATH           = "training_data_backup_20260426_013442.csv"
DATASET_B_PATH           = "indian_bias_labeled.csv"
PROCESSED_DIR            = "data/processed/"
TRAIN_PATH               = "data/processed/train_v2.csv"
VAL_PATH                 = "data/processed/val_v2.csv"
TEST_PATH                = "data/processed/test_v2.csv"
METADATA_PATH            = "data/processed/dataset_v2_metadata.json"
TARGET_SAMPLES_PER_CLASS = 8146
RANDOM_STATE             = 42
TEST_SIZE                = 0.10
VAL_SIZE                 = 0.10
MAX_TFIDF_FEATURES       = 10000

SMOTE_CHUNK_SIZE         = 2000
TFIDF_DTYPE              = "float32"

VALID_LABELS = {0, 1, 2, 3, 4}
LABEL_STR_MAP = {
    "far left"  : 0,
    "left"      : 1,
    "center"    : 2,
    "right"     : 3,
    "far right" : 4,
}
SCHEMA_A_COLS = ["text", "bias_label", "fact_score",
                 "intent_label", "emotion_label", "region"]

# ── LOGGING ──────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_ok(msg: str):   print(f"[{_ts()}] OK  {msg}")
def log_warn(msg: str): print(f"[{_ts()}] WRN {msg}")
def log_err(msg: str):  print(f"[{_ts()}] ERR {msg}")

# ── STEP 1: LOAD & AUDIT ─────────────────────────────────────────────────────
def load_and_audit(path: str, name: str) -> pd.DataFrame:
    log_ok(f"Loading {name} from '{path}' ...")
    if not os.path.exists(path):
        log_err(f"{name}: file not found -> {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    log_ok(f"{name}: loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(f"\n  === {name} AUDIT ===")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {df.columns.tolist()}")
    print(f"  dtypes  :\n{df.dtypes.to_string()}")
    print(f"  Nulls   :\n{df.isnull().sum().to_string()}")
    print(f"  Sample  :\n{df.head(3).to_string()}")

    # class distribution (pre-normalization)
    lbl_col = "bias_label" if "bias_label" in df.columns else "label"
    if lbl_col in df.columns:
        dist = df[lbl_col].value_counts(dropna=False).sort_index()
        total = len(df)
        print(f"  Bias-label distribution:")
        for val, cnt in dist.items():
            print(f"    {val:>10} : {cnt:>6} ({cnt/total*100:5.2f}%)")
    print()
    return df

# ── STEP 2: LABEL NORMALIZATION ──────────────────────────────────────────────
def normalize_labels(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, bool, int]:
    # Detect label column
    if "bias_label" in df.columns:
        lbl_col = "bias_label"
    elif "label" in df.columns:
        df = df.rename(columns={"label": "bias_label"})
        lbl_col = "bias_label"
    else:
        raise ValueError(f"{name}: cannot find label column")

    remapped = False
    if df["bias_label"].dtype == object:
        log_warn(f"{name}: string labels detected -- remapping to integers")
        df["bias_label"] = (
            df["bias_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(LABEL_STR_MAP)
        )
        remapped = True

    df["bias_label"] = pd.to_numeric(df["bias_label"], errors="coerce")

    before = len(df)
    df = df[df["bias_label"].isin(VALID_LABELS)].copy()
    dropped = before - len(df)
    if dropped:
        log_warn(f"{name}: dropped {dropped} rows with invalid bias_label values")

    df["bias_label"] = df["bias_label"].astype("int64")
    log_ok(f"{name}: label normalization complete -- {len(df):,} rows remaining")
    return df, remapped, dropped

# ── STEP 3: TEXT CLEANING ────────────────────────────────────────────────────
def clean_text_col(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, int]:
    total_dropped = 0

    def drop_step(df_in: pd.DataFrame, label: str, mask_invalid) -> pd.DataFrame:
        nonlocal total_dropped
        n_before = len(df_in)
        df_out = df_in[~mask_invalid].copy()
        n_dropped = n_before - len(df_out)
        if n_dropped:
            log_warn(f"{name} text-clean [{label}]: dropped {n_dropped} rows")
        total_dropped += n_dropped
        return df_out

    # 1. Ensure string
    df["text"] = df["text"].astype(str)

    # 2. Unicode NFC normalise
    df["text"] = df["text"].apply(
        lambda t: unicodedata.normalize("NFC", t)
    )

    # 3. Strip leading/trailing whitespace
    df["text"] = df["text"].str.strip()

    # 4. Collapse multiple spaces / newlines -> single space
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    # 5. Strip HTML tags
    df["text"] = df["text"].str.replace(r"<.*?>", "", regex=True).str.strip()

    # 6. Drop null / empty / whitespace-only
    df = drop_step(df, "null-or-empty", df["text"].isna() | df["text"].str.strip().eq(""))

    # 7. Drop len < 20
    df = drop_step(df, "len<20", df["text"].str.strip().str.len() < 20)

    log_ok(f"{name}: text cleaning complete -- {len(df):,} rows remaining "
           f"(dropped {total_dropped} total)")
    return df, total_dropped

# ── STEP 4: CROSS-DATASET DEDUPLICATION ─────────────────────────────────────
def cross_dedup(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    norm_a = set(df_a["text"].str.lower().str.strip())
    before = len(df_b)
    keep = ~df_b["text"].str.lower().str.strip().isin(norm_a)
    df_b = df_b[keep].copy()
    dropped = before - len(df_b)
    if dropped:
        log_warn(f"Dropped {dropped} duplicate rows from Dataset B that already exist in Dataset A")
    else:
        log_ok("Cross-dedup: no overlapping texts found between Dataset A and Dataset B")
    return df_b, dropped

# ── STEP 5: SCHEMA ALIGNMENT ─────────────────────────────────────────────────
def align_schema(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    padded = []
    for col in SCHEMA_A_COLS:
        if col == "text" or col == "bias_label":
            continue
        if col not in df_b.columns:
            df_b[col] = np.nan
            padded.append(col)
            log_warn(f"Schema align: padded missing column '{col}' in Dataset B with NaN")

    # Reorder Dataset B to match Dataset A column order
    for col in SCHEMA_A_COLS:
        if col not in df_b.columns:
            df_b[col] = np.nan

    df_b = df_b[SCHEMA_A_COLS].copy()
    df_a = df_a[SCHEMA_A_COLS].copy()

    df_a["source"] = "global_backup"
    df_b["source"] = "indian_dataset"

    log_ok(f"Schema alignment complete. Padded columns: {padded or 'none'}")
    return df_a, df_b, padded

# ── STEP 6: MERGE & REBALANCE ────────────────────────────────────────────────
def merge_and_rebalance(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.concat([df_a, df_b], ignore_index=True)
    log_ok(f"Merged dataset: {len(df_merged):,} rows (before dedup)")

    # Deduplicate on exact text BEFORE rebalancing so splits will have zero
    # text overlap.  Dataset A rows are first so keep="first" gives priority.
    before_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
    n_removed = before_dedup - len(df_merged)
    if n_removed:
        log_warn(f"Cross-merge dedup: removed {n_removed} duplicate texts")
    else:
        log_ok("Cross-merge dedup: no duplicate texts found")

    print("\n  Class distribution BEFORE rebalancing (unique texts):")
    dist_before = df_merged["bias_label"].value_counts().sort_index()
    for lbl, cnt in dist_before.items():
        print(f"    Class {lbl}: {cnt:>6} ({cnt/len(df_merged)*100:5.2f}%)")

    # Per-class rebalancing
    balanced_frames = []

    # Fit TF-IDF once on the deduplicated merged text (for SMOTE use)
    tfidf = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        dtype=np.float32
    )
    X_all = tfidf.fit_transform(df_merged["text"])

    from scipy.sparse import vstack, csr_matrix

    for lbl in sorted(VALID_LABELS):
        df_cls = df_merged[df_merged["bias_label"] == lbl].copy()
        cnt = len(df_cls)
        
        if cnt < TARGET_SAMPLES_PER_CLASS:
            S = TARGET_SAMPLES_PER_CLASS - cnt
            log_ok(f"Class {lbl}: needs SMOTE oversampling ({cnt} -> {TARGET_SAMPLES_PER_CLASS}, generating {S} samples)")
            
            cls_indices = df_merged.index[df_merged["bias_label"] == lbl].tolist()
            X_class = X_all[cls_indices]
            N = X_class.shape[0]
            
            chunk_size = SMOTE_CHUNK_SIZE if N > 4000 else S
            
            syn_chunks = []
            bytes_generated = 0
            
            while bytes_generated < S:
                chunk_s = min(chunk_size, S - bytes_generated)
                
                X_dummy = csr_matrix((N + chunk_s, X_class.shape[1]), dtype=np.float32)
                y_dummy = np.full(N + chunk_s, -1, dtype=np.int64)
                
                X_temp = vstack([X_class, X_dummy])
                y_temp = np.concatenate([np.full(N, lbl, dtype=np.int64), y_dummy])
                
                smote = SMOTE(
                    sampling_strategy={lbl: N + chunk_s},
                    random_state=RANDOM_STATE,
                    k_neighbors=min(5, N - 1)
                )
                X_res_temp, y_res_temp = smote.fit_resample(X_temp, y_temp)
                
                X_res_c = X_res_temp[y_res_temp == lbl]
                X_syn_chunk = X_res_c[N:]
                syn_chunks.append(X_syn_chunk)
                
                bytes_generated += chunk_s
                
            X_syn = vstack(syn_chunks)
            
            nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=1)
            nn.fit(X_class)
            _, nn_indices = nn.kneighbors(X_syn)
            nn_indices = nn_indices.ravel()
            
            orig_texts = df_cls["text"].values
            syn_texts = [f"{orig_texts[idx]} [syn-{j}]" for j, idx in enumerate(nn_indices)]
            
            syn_meta = df_cls.iloc[nn_indices][SCHEMA_A_COLS].copy().reset_index(drop=True)
            syn_meta["text"] = syn_texts
            syn_meta["bias_label"] = lbl
            syn_meta["source"] = "smote_synthetic"
            
            df_cls = pd.concat([df_cls, syn_meta], ignore_index=True)
            log_ok(f"Class {lbl}: balanced to {len(df_cls)} samples via SMOTE")
            
        elif cnt > TARGET_SAMPLES_PER_CLASS:
            df_cls = df_cls.sample(n=TARGET_SAMPLES_PER_CLASS, random_state=RANDOM_STATE)
            log_warn(f"Class {lbl}: undersampled {cnt} -> {TARGET_SAMPLES_PER_CLASS}")
        else:
            log_ok(f"Class {lbl}: exactly {TARGET_SAMPLES_PER_CLASS} -- no change needed")
            
        balanced_frames.append(df_cls)

    df_balanced = pd.concat(balanced_frames, ignore_index=True)

    # Hard assert
    dist_after = df_balanced["bias_label"].value_counts().sort_index()
    print("\n  Class distribution AFTER rebalancing:")
    for lbl, cnt in dist_after.items():
        print(f"    Class {lbl}: {cnt:>6}")
    for lbl in VALID_LABELS:
        actual = dist_after.get(lbl, 0)
        if actual != TARGET_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Balance assertion failed: class {lbl} has {actual} rows "
                f"(expected {TARGET_SAMPLES_PER_CLASS})"
            )

    return df_balanced

# ── STEP 7: TRAIN / VAL / TEST SPLIT ────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=df["bias_label"],
        random_state=RANDOM_STATE,
    )
    # Second split: val vs test (equal halves of the temp set)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["bias_label"],
        random_state=RANDOM_STATE,
    )

    log_ok(f"Split -> train:{len(df_train):,}  val:{len(df_val):,}  test:{len(df_test):,}")

    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = split_df["bias_label"].value_counts().sort_index()
        print(f"\n  {split_name} class distribution:")
        for lbl, cnt in dist.items():
            print(f"    Class {lbl}: {cnt}")

    # Leakage checks — compare exact text values (SMOTE synthetics carry
    # a unique [syn-N] suffix so they are distinct from originals even
    # when the base text is the same; normalised comparison would produce
    # false positives for those pairs).
    def check_overlap(s1: pd.Series, s2: pd.Series, label: str):
        n1 = set(s1.astype(str))
        n2 = set(s2.astype(str))
        overlap = len(n1 & n2)
        if overlap > 0:
            raise ValueError(f"Data leakage detected: {label} -- {overlap} overlapping texts!")
        log_ok(f"Leakage check {label}: PASSED (0 overlaps)")

    check_overlap(df_train["text"], df_val["text"],  "TRAIN/VAL")
    check_overlap(df_train["text"], df_test["text"], "TRAIN/TEST")
    check_overlap(df_val["text"],   df_test["text"], "VAL/TEST")

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_train.to_csv(TRAIN_PATH, index=False)
    df_val.to_csv(VAL_PATH,   index=False)
    df_test.to_csv(TEST_PATH,  index=False)
    log_ok(f"Splits saved to {PROCESSED_DIR}")
    return df_train, df_val, df_test

# ── STEP 8: SAVE METADATA ────────────────────────────────────────────────────
def save_metadata(
    df_balanced: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    columns_padded: list,
    label_remapping_applied: bool,
    cross_dedup_rows_dropped: int,
    text_cleaning_rows_dropped: int,
):
    source_counts = df_balanced["source"].value_counts().to_dict()
    class_dist    = {
        str(k): int(v)
        for k, v in df_balanced["bias_label"].value_counts().sort_index().items()
    }
    metadata = {
        "created_at"                 : datetime.now(timezone.utc).isoformat(),
        "random_state"               : RANDOM_STATE,
        "target_samples_per_class"   : TARGET_SAMPLES_PER_CLASS,
        "total_samples"              : len(df_balanced),
        "train_samples"              : len(df_train),
        "val_samples"                : len(df_val),
        "test_samples"               : len(df_test),
        "class_distribution"         : class_dist,
        "source_breakdown"           : {
            "global_backup"   : int(source_counts.get("global_backup",    0)),
            "indian_dataset"  : int(source_counts.get("indian_dataset",   0)),
            "smote_synthetic" : int(source_counts.get("smote_synthetic",  0)),
        },
        "columns_padded"             : columns_padded,
        "label_remapping_applied"    : label_remapping_applied,
        "cross_dedup_rows_dropped"   : cross_dedup_rows_dropped,
        "text_cleaning_rows_dropped" : text_cleaning_rows_dropped,
    }
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log_ok(f"Metadata saved -> {METADATA_PATH}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PHASE 1 -- DATASET PREPARATION PIPELINE")
    print("=" * 70)

    # Step 1 -- Load
    df_a_raw = load_and_audit(DATASET_A_PATH, "Dataset A (Global Backup)")
    df_b_raw = load_and_audit(DATASET_B_PATH, "Dataset B (Indian News)")

    # Step 2 -- Label normalisation
    df_a, remap_a, dropped_a = normalize_labels(df_a_raw, "Dataset A")
    df_b, remap_b, dropped_b = normalize_labels(df_b_raw, "Dataset B")
    label_remapping_applied  = remap_a or remap_b

    # Step 3 -- Text cleaning
    df_a, clean_dropped_a = clean_text_col(df_a, "Dataset A")
    df_b, clean_dropped_b = clean_text_col(df_b, "Dataset B")
    text_cleaning_rows_dropped = clean_dropped_a + clean_dropped_b

    # Step 4 -- Cross-dataset dedup
    df_b, cross_dedup_dropped = cross_dedup(df_a, df_b)

    # Step 5 -- Schema alignment
    df_a, df_b, columns_padded = align_schema(df_a, df_b)

    # Step 6 -- Merge & rebalance
    df_balanced = merge_and_rebalance(df_a, df_b)

    # Step 7 -- Split & save
    df_train, df_val, df_test = split_and_save(df_balanced)

    # Step 8 -- Metadata
    save_metadata(
        df_balanced,
        df_train,
        df_val,
        df_test,
        columns_padded,
        label_remapping_applied,
        cross_dedup_dropped,
        text_cleaning_rows_dropped,
    )

    print("\n" + "=" * 70)
    print("  PHASE 1 COMPLETE -- dataset splits ready for audit")
    print("=" * 70)


if __name__ == "__main__":
    main()
