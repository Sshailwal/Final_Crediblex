"""
audit_dataset_v2.py
Phase 2 -- Pre-Training Data Audit
Political Bias Classification (5 classes: 0=Far Left ... 4=Far Right)

Usage:
    python audit_dataset_v2.py          # standalone
    from audit_dataset_v2 import run_audit; run_audit()  # module import
"""

import os
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── PATH CONSTANTS ───────────────────────────────────────────────────────────
TRAIN_PATH    = "data/processed/train_v2.csv"
VAL_PATH      = "data/processed/val_v2.csv"
TEST_PATH     = "data/processed/test_v2.csv"
METADATA_PATH = "data/processed/dataset_v2_metadata.json"

EXPECTED_COLS    = ["text", "bias_label", "fact_score",
                    "intent_label", "emotion_label", "region", "source"]
EXPECTED_CLASSES = {0, 1, 2, 3, 4}
VALID_SOURCES    = {"global_backup", "indian_dataset", "smote_synthetic"}

# ── CUSTOM EXCEPTION ─────────────────────────────────────────────────────────
class AuditFailureError(Exception):
    """Raised when one or more audit checks fail."""
    pass

# ── LOGGING ──────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _pass(check_id: str, msg: str = "") -> bool:
    suffix = f" -- {msg}" if msg else ""
    print(f"[{_ts()}] PASS [{check_id}]{suffix}")
    return True

def _fail(check_id: str, msg: str = "") -> bool:
    suffix = f" -- {msg}" if msg else ""
    print(f"[{_ts()}] FAIL [{check_id}]{suffix}")
    return False

# ── AUDIT RUNNER ─────────────────────────────────────────────────────────────
def run_audit() -> bool:
    failures: list[str] = []
    results: dict[str, dict] = {}  # block_name -> {passed, total}

    def record(block: str, check_id: str, passed: bool):
        if block not in results:
            results[block] = {"passed": 0, "total": 0}
        results[block]["total"] += 1
        if passed:
            results[block]["passed"] += 1
        else:
            failures.append(check_id)

    # ── Load files (best-effort; FS checks handle missing) ──────────────────
    def safe_load(path: str):
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def safe_json(path: str):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    train_df = safe_load(TRAIN_PATH)
    val_df   = safe_load(VAL_PATH)
    test_df  = safe_load(TEST_PATH)
    meta     = safe_json(METADATA_PATH)

    # ────────────────────────────────────────────────────────────────────────
    #  BLOCK 1 -- FILE SYSTEM
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] --- BLOCK 1: FILE SYSTEM ---")

    ok = os.path.exists(TRAIN_PATH) and os.path.getsize(TRAIN_PATH) > 0
    record("BLOCK 1", "FS-01", _pass("FS-01", "train_v2.csv exists") if ok else _fail("FS-01", "train_v2.csv missing or empty"))

    ok = os.path.exists(VAL_PATH) and os.path.getsize(VAL_PATH) > 0
    record("BLOCK 1", "FS-02", _pass("FS-02", "val_v2.csv exists") if ok else _fail("FS-02", "val_v2.csv missing or empty"))

    ok = os.path.exists(TEST_PATH) and os.path.getsize(TEST_PATH) > 0
    record("BLOCK 1", "FS-03", _pass("FS-03", "test_v2.csv exists") if ok else _fail("FS-03", "test_v2.csv missing or empty"))

    ok = meta is not None
    record("BLOCK 1", "FS-04", _pass("FS-04", "metadata.json parsed") if ok else _fail("FS-04", "metadata.json missing or invalid JSON"))

    # FS-05: metadata row counts match actual splits
    if meta is not None and all(d is not None for d in [train_df, val_df, test_df]):
        match = (
            meta.get("train_samples") == len(train_df)
            and meta.get("val_samples")   == len(val_df)
            and meta.get("test_samples")  == len(test_df)
        )
        record("BLOCK 1", "FS-05",
               _pass("FS-05", "metadata row counts match splits") if match
               else _fail("FS-05",
                          f"mismatch: meta={meta.get('train_samples')}/{meta.get('val_samples')}/{meta.get('test_samples')} "
                          f"actual={len(train_df)}/{len(val_df)}/{len(test_df)}"))
    else:
        record("BLOCK 1", "FS-05", _fail("FS-05", "cannot verify -- missing files"))

    # ────────────────────────────────────────────────────────────────────────
    #  BLOCK 2 -- SCHEMA
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] --- BLOCK 2: SCHEMA ---")

    if all(d is not None for d in [train_df, val_df, test_df]):
        same_cols = (list(train_df.columns) == list(val_df.columns) == list(test_df.columns))
        record("BLOCK 2", "SC-01",
               _pass("SC-01", "all splits share identical column order") if same_cols
               else _fail("SC-01", "column mismatch between splits"))

        ok = (train_df["bias_label"].dtype == "int64"
              and val_df["bias_label"].dtype  == "int64"
              and test_df["bias_label"].dtype  == "int64")
        record("BLOCK 2", "SC-02",
               _pass("SC-02", "bias_label is int64 in all splits") if ok
               else _fail("SC-02", "bias_label dtype mismatch"))

        has_src = all("source" in d.columns for d in [train_df, val_df, test_df])
        record("BLOCK 2", "SC-03",
               _pass("SC-03", "source column present in all splits") if has_src
               else _fail("SC-03", "source column missing in one or more splits"))

        all_sources = set(
            pd.concat([train_df["source"], val_df["source"], test_df["source"]]).unique()
        ) if has_src else set()
        ok = all_sources.issubset(VALID_SOURCES)
        record("BLOCK 2", "SC-04",
               _pass("SC-04", f"source values are valid: {all_sources}") if ok
               else _fail("SC-04", f"unknown source values: {all_sources - VALID_SOURCES}"))
    else:
        for cid in ["SC-01", "SC-02", "SC-03", "SC-04"]:
            record("BLOCK 2", cid, _fail(cid, "cannot check -- missing files"))

    # ────────────────────────────────────────────────────────────────────────
    #  BLOCK 3 -- DATASET INTEGRITY
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] --- BLOCK 3: DATASET INTEGRITY ---")

    if all(d is not None for d in [train_df, val_df, test_df]):
        # DI-01 null text
        null_text = sum(d["text"].isnull().sum() for d in [train_df, val_df, test_df])
        record("BLOCK 3", "DI-01",
               _pass("DI-01", "zero null text values") if null_text == 0
               else _fail("DI-01", f"{null_text} null text values found"))

        # DI-02 null bias_label
        null_lbl = sum(d["bias_label"].isnull().sum() for d in [train_df, val_df, test_df])
        record("BLOCK 3", "DI-02",
               _pass("DI-02", "zero null bias_label values") if null_lbl == 0
               else _fail("DI-02", f"{null_lbl} null bias_label values found"))

        # DI-03 all labels in {0..4}
        all_lbls = pd.concat([d["bias_label"] for d in [train_df, val_df, test_df]])
        unknown = set(all_lbls.unique()) - EXPECTED_CLASSES
        record("BLOCK 3", "DI-03",
               _pass("DI-03", "all bias_label values within {0,1,2,3,4}") if not unknown
               else _fail("DI-03", f"unknown labels found: {unknown}"))

        # DI-04 zero empty strings
        def count_empty(df):
            return df["text"].astype(str).str.strip().eq("").sum()
        n_empty = sum(count_empty(d) for d in [train_df, val_df, test_df])
        record("BLOCK 3", "DI-04",
               _pass("DI-04", "zero empty text strings") if n_empty == 0
               else _fail("DI-04", f"{n_empty} empty text values found"))

        # DI-05 no text < 20 chars
        def count_short(df):
            return (df["text"].astype(str).str.strip().str.len() < 20).sum()
        n_short = sum(count_short(d) for d in [train_df, val_df, test_df])
        record("BLOCK 3", "DI-05",
               _pass("DI-05", "zero texts shorter than 20 chars") if n_short == 0
               else _fail("DI-05", f"{n_short} texts with len < 20 chars"))

        # DI-06 no duplicate text in training set
        n_dupes = train_df.duplicated(subset=["text"]).sum()
        record("BLOCK 3", "DI-06",
               _pass("DI-06", "zero duplicate texts in train set") if n_dupes == 0
               else _fail("DI-06", f"{n_dupes} duplicate texts in train set"))

        # DI-07 metadata total == actual total
        if meta is not None:
            expected_total = meta.get("total_samples", -1)
            actual_total   = len(train_df) + len(val_df) + len(test_df)
            ok = expected_total == actual_total
            record("BLOCK 3", "DI-07",
                   _pass("DI-07", f"metadata total {expected_total} matches actual {actual_total}") if ok
                   else _fail("DI-07", f"metadata total {expected_total} != actual {actual_total}"))
        else:
            record("BLOCK 3", "DI-07", _fail("DI-07", "cannot verify -- metadata missing"))
    else:
        for cid in ["DI-01","DI-02","DI-03","DI-04","DI-05","DI-06","DI-07"]:
            record("BLOCK 3", cid, _fail(cid, "cannot check -- missing files"))

    # ────────────────────────────────────────────────────────────────────────
    #  BLOCK 4 -- CLASS BALANCE
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] --- BLOCK 4: CLASS BALANCE ---")

    if all(d is not None for d in [train_df, val_df, test_df]):
        total = len(train_df) + len(val_df) + len(test_df)
        tr_cnt = train_df["bias_label"].value_counts().sort_index()
        val_cnt = val_df["bias_label"].value_counts().sort_index()
        tst_cnt = test_df["bias_label"].value_counts().sort_index()

        # CB-01 train has all 5 classes
        record("BLOCK 4", "CB-01",
               _pass("CB-01", "train contains all 5 classes") if set(tr_cnt.index) == EXPECTED_CLASSES
               else _fail("CB-01", f"train missing classes: {EXPECTED_CLASSES - set(tr_cnt.index)}"))

        # CB-02 train balanced within ±1
        max_diff = tr_cnt.max() - tr_cnt.min()
        record("BLOCK 4", "CB-02",
               _pass("CB-02", f"train balanced (max_diff={max_diff})") if max_diff <= 1
               else _fail("CB-02", f"train imbalanced -- max_diff={max_diff}"))

        # CB-03 val has all 5 classes
        record("BLOCK 4", "CB-03",
               _pass("CB-03", "val contains all 5 classes") if set(val_cnt.index) >= EXPECTED_CLASSES
               else _fail("CB-03", f"val missing classes: {EXPECTED_CLASSES - set(val_cnt.index)}"))

        # CB-04 test has all 5 classes
        record("BLOCK 4", "CB-04",
               _pass("CB-04", "test contains all 5 classes") if set(tst_cnt.index) >= EXPECTED_CLASSES
               else _fail("CB-04", f"test missing classes: {EXPECTED_CLASSES - set(tst_cnt.index)}"))

        # CB-05/06/07 split ratios
        def check_ratio(split_len, expected_pct, tol=0.005, cid="", label=""):
            actual_pct = split_len / total
            ok = abs(actual_pct - expected_pct) <= tol
            msg = f"{label} split ratio {actual_pct:.4f} (expected ~{expected_pct:.2f} ±{tol})"
            return _pass(cid, msg) if ok else _fail(cid, msg)

        record("BLOCK 4", "CB-05", check_ratio(len(train_df), 0.80, cid="CB-05", label="train"))
        record("BLOCK 4", "CB-06", check_ratio(len(val_df),   0.10, cid="CB-06", label="val"))
        record("BLOCK 4", "CB-07", check_ratio(len(test_df),  0.10, cid="CB-07", label="test"))
    else:
        for cid in ["CB-01","CB-02","CB-03","CB-04","CB-05","CB-06","CB-07"]:
            record("BLOCK 4", cid, _fail(cid, "cannot check -- missing files"))

    # ────────────────────────────────────────────────────────────────────────
    #  BLOCK 5 -- DATA LEAKAGE
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] --- BLOCK 5: DATA LEAKAGE ---")

    if all(d is not None for d in [train_df, val_df, test_df]):
        def norm_set(df):
            return set(df["text"].astype(str).str.lower().str.strip())

        tr_set  = norm_set(train_df)
        val_set = norm_set(val_df)
        tst_set = norm_set(test_df)

        n = len(tr_set & val_set)
        record("BLOCK 5", "DL-01",
               _pass("DL-01", "zero train/val text overlap") if n == 0
               else _fail("DL-01", f"{n} overlapping texts between train and val"))

        n = len(tr_set & tst_set)
        record("BLOCK 5", "DL-02",
               _pass("DL-02", "zero train/test text overlap") if n == 0
               else _fail("DL-02", f"{n} overlapping texts between train and test"))

        n = len(val_set & tst_set)
        record("BLOCK 5", "DL-03",
               _pass("DL-03", "zero val/test text overlap") if n == 0
               else _fail("DL-03", f"{n} overlapping texts between val and test"))
    else:
        for cid in ["DL-01","DL-02","DL-03"]:
            record("BLOCK 5", cid, _fail(cid, "cannot check -- missing files"))

    # ── FINAL REPORT ─────────────────────────────────────────────────────────
    block_order  = ["BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4", "BLOCK 5"]
    block_labels = {
        "BLOCK 1": "FILE SYSTEM        ",
        "BLOCK 2": "SCHEMA             ",
        "BLOCK 3": "DATASET INTEGRITY  ",
        "BLOCK 4": "CLASS BALANCE      ",
        "BLOCK 5": "DATA LEAKAGE       ",
    }
    total_passed = sum(r["passed"] for r in results.values())
    total_checks = sum(r["total"]  for r in results.values())
    verdict_ok   = len(failures) == 0

    width = 64
    print("\n" + "=" * width)
    print("  PHASE 2 -- PRE-TRAINING AUDIT REPORT")
    print("=" * width)
    for blk in block_order:
        r = results.get(blk, {"passed": 0, "total": 0})
        label = block_labels.get(blk, blk)
        print(f"  {label} : {r['passed']} / {r['total']} passed")
    print("-" * width)
    print(f"  TOTAL                          : {total_passed} / {total_checks} passed")
    failed_str = ", ".join(failures) if failures else "NONE"
    print(f"  FAILED CHECKS                  : {failed_str}")
    print("-" * width)
    if verdict_ok:
        print(f"  VERDICT : ALL CHECKS PASSED -- SAFE TO PROCEED TO PHASE 3")
    else:
        print(f"  VERDICT : AUDIT FAILED -- DO NOT PROCEED TO TRAINING")
    print("=" * width + "\n")

    if not verdict_ok:
        raise AuditFailureError(
            f"Audit failed: {len(failures)} check(s) did not pass: {failures}"
        )

    return True


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run_audit()
    except AuditFailureError as e:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERR {e}")
        sys.exit(1)
