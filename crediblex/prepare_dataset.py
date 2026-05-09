# coding: utf-8
"""
prepare_dataset.py - CredibleX Dataset Builder v6

Implementation Plan v1 — US-Only, 3-Class Bias, Proper Sentinel Labels.

Key fixes over v5:
  - Sentinel labels: rows only carry labels for their OWN task.
      bias_label  = -1  for non-bias rows (LIAR, FakeNews, GoEmotions)
      fact_score  = -1.0 for non-fact rows (BABE, GoEmotions emotion-only)
      intent_label = -1 for non-intent rows (BABE, LIAR, GoEmotions)
      emotion_label = all-zeros vector for non-emotion rows (BABE, LIAR, FakeNews)
  - Neutral cap lowered to 600 (20% of 3,000 emotion rows).
  - Each non-bias task capped at 3,000 rows to reduce task-sampling imbalance.
  - BIAS_TARGET_PER_CLASS raised to 742 (min of Left/Right raw counts, use all).

Sources (Training):
  1. mediabiasgroup/BABE     — bias only  (bias=0/1/2, fact=-1, intent=-1, emo=zeros)
  2. UKPLab/liar             — fact only  (bias=-1, fact=0.0/1.0, intent=-1, emo=zeros)
  3. GonzaloA/fake_news      — intent only (bias=-1, fact=proxy, intent=0/2, emo=zeros)
  4. go_emotions             — emotion only (bias=-1, fact=proxy, intent=-1, emo=one-hot)

Bias Labels:   0=Left, 1=Center, 2=Right  (-1=no label)
Intent Labels: 0=News, 1=Satire/Fake      (-1=no label)
Emotion:       7-class Ekman one-hot      (all-zeros = no label)
Fact:          0.0–1.0 regression         (-1.0 = no label)
"""

import os, sys, argparse, datetime, shutil, json, random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Constants ─────────────────────────────────────────────────────────────────
BIAS_NAMES   = {0: "Left", 1: "Center", 2: "Right"}
EKMAN_NAMES  = {0: "Joy", 1: "Sadness", 2: "Anger", 3: "Fear",
                4: "Surprise", 5: "Disgust", 6: "Neutral"}
N_EKMAN      = 7

# ── Task row caps ─────────────────────────────────────────────────────────────
# Caps each non-bias task at 3,000 to reduce the 8x task-sampling imbalance.
TASK_ROW_CAP     = 3000

# ── Bias class target ─────────────────────────────────────────────────────────
# Raw BABE counts: Left~742, Center~1378, Right~881.
# The strict sentence-level 'type' filtering leaves exactly 549 'left' rows.
BIAS_TARGET_PER_CLASS = 549

# ── GoEmotions 28-class → 7 Ekman mapping ────────────────────────────────────
# Unmapped GoEmotions classes are SKIPPED (not funnelled to Neutral).
GO_TO_EKMAN = {
    # Joy
    1: 0, 5: 0, 8: 0, 9: 0, 12: 0, 17: 0, 20: 0, 23: 0,
    # Sadness
    16: 1, 15: 1, 10: 1, 6: 1, 7: 1,
    # Anger
    2: 2, 14: 2, 4: 2,
    # Fear
    13: 3, 22: 3,
    # Surprise
    24: 4, 3: 4,
    # Disgust
    11: 5, 25: 5,
    # Neutral (explicit GoEmotions label 27)
    27: 6,
}

# ── Factuality proxy for emotion rows (not used for regression training) ──────
EMO_FACT_PROXY = {0: 0.65, 1: 0.45, 2: 0.30, 3: 0.40, 4: 0.55, 5: 0.35, 6: 0.60}

# ── LIAR binary label mapping ─────────────────────────────────────────────────
# UKPLab/liar: labels=0 → 'true statement' → 1.0; labels=1 → 'false statement' → 0.0
LIAR_FACT_BINARY = {0: 1.0, 1: 0.0}

# ── BABE outlet lean ──────────────────────────────────────────────────────────
BABE_LEFT_OUTLETS  = {"alternet", "msnbc", "huffpost", "huffington post",
                       "the daily beast", "daily beast", "the guardian"}
BABE_RIGHT_OUTLETS = {"breitbart", "federalist", "the federalist", "fox news",
                       "theblaze", "the blaze", "daily stormer", "new york post"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_multihot(ekman_idx, n=N_EKMAN):
    """
    Return a JSON multi-hot vector.
    ekman_idx = -1  → all-zeros sentinel (row has NO emotion label).
    """
    vec = [0.0] * n
    if 0 <= ekman_idx < n:
        vec[ekman_idx] = 1.0
    return json.dumps(vec)


def _row(text, bias, fact, intent, ekman_idx, region="US", source="other"):
    """
    Construct a dataset row with proper sentinel values.
    Pass bias=-1, fact=-1.0, intent=-1, or ekman_idx=-1 for tasks
    that this row does NOT carry a label for.
    """
    return {
        "text":           str(text).strip()[:3000],
        "bias_label":     int(bias),          # -1 = no bias label
        "fact_score":     float(fact),        # -1.0 = no factuality label
        "intent_label":   int(intent),        # -1 = no intent label
        "emotion_label":  _make_multihot(ekman_idx),  # all-zeros = no emotion label
        "region":         str(region),
        "_primary_ekman": int(ekman_idx),     # -1 for non-emotion rows
        "_source":        str(source),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: BABE — bias-only rows
# bias=0/1/2  |  fact=-1.0 (no fact label)  |  intent=-1  |  emo=zeros
# ─────────────────────────────────────────────────────────────────────────────
def load_babe():
    """Returns dict {cls: [rows]} for undersampling."""
    buckets = {0: [], 1: [], 2: []}
    skipped = 0
    try:
        print("\n[1/4] BABE (bias-only rows, outlet+content strategy)...")
        ds = load_dataset("mediabiasgroup/babe", split="train")
        for r in tqdm(ds, desc="  BABE"):
            text      = str(r.get("text", "")).strip()
            if len(text) < 20:
                continue
            is_biased = (int(r.get("label", 0)) == 1)
            b_type    = str(r.get("type", "")).lower().strip()

            if is_biased:
                if b_type == "left":
                    bias = 0
                elif b_type == "right":
                    bias = 2
                else:
                    skipped += 1
                    continue
            else:
                bias = 1  # not-biased → Center

            # fact=-1.0, intent=-1, ekman=-1 — bias-only row
            buckets[bias].append(_row(text, bias, -1.0, -1, -1, "US", "babe"))

        for cls, rows in buckets.items():
            print("  Raw {}: {:,}".format(BIAS_NAMES[cls], len(rows)))
        print("  Skipped (biased, unknown outlet): {:,}".format(skipped))
    except Exception as e:
        print("  [SKIP] BABE: {}".format(e))
    return buckets


def balance_bias(buckets):
    """Undersample each class to BIAS_TARGET_PER_CLASS. No oversampling."""
    print("\n[*] Balancing bias classes (target={:,} each)...".format(BIAS_TARGET_PER_CLASS))
    all_rows = []
    for cls in range(3):
        bucket = buckets.get(cls, [])
        n = len(bucket)
        if n == 0:
            print("  [WARN] Class {} ({}) = 0 rows".format(cls, BIAS_NAMES[cls]))
            continue
        if n > BIAS_TARGET_PER_CLASS:
            random.seed(42)
            bucket = random.sample(bucket, BIAS_TARGET_PER_CLASS)
            print("  {} ({}): {:,} → {:,} (undersampled)".format(BIAS_NAMES[cls], cls, n, len(bucket)))
        else:
            print("  {} ({}): {:,} (all kept)".format(BIAS_NAMES[cls], cls, n))
        all_rows.extend(bucket)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: LIAR — factuality-only rows
# bias=-1  |  fact=0.0 or 1.0  |  intent=-1  |  emo=zeros
# ─────────────────────────────────────────────────────────────────────────────
def load_liar(limit=TASK_ROW_CAP):
    rows = []
    try:
        print("\n[2/4] UKPLab/liar (fact-only rows, cap={:,})...".format(limit))
        ds = load_dataset("UKPLab/liar")
        for split in ("train", "validation", "test"):
            if split not in ds:
                continue
            for r in tqdm(ds[split], desc="  LIAR-" + split):
                if len(rows) >= limit:
                    break
                text = str(r.get("text", "")).strip()
                if len(text) < 10:
                    continue
                raw = r.get("labels", r.get("label", None))
                if raw is None:
                    continue
                try:
                    fact = LIAR_FACT_BINARY.get(int(raw), 0.5)
                except (TypeError, ValueError):
                    fact = 0.5
                # bias=-1, intent=-1, ekman=-1 — fact-only row
                rows.append(_row(text, -1, fact, -1, -1, "US", "liar"))
        print("  [OK] LIAR: {:,} rows  true={:,}  false={:,}".format(
            len(rows),
            sum(1 for r in rows if r["fact_score"] == 1.0),
            sum(1 for r in rows if r["fact_score"] == 0.0),
        ))
    except Exception as e:
        print("  [SKIP] LIAR: {}".format(e))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: GonzaloA/fake_news — intent-only rows
# bias=-1  |  fact=proxy  |  intent=0/2  |  emo=zeros
# ─────────────────────────────────────────────────────────────────────────────
def load_fakenews(limit=TASK_ROW_CAP):
    rows = []
    try:
        print("\n[3/4] GonzaloA/fake_news (intent-only rows, cap={:,})...".format(limit))
        ds = load_dataset("GonzaloA/fake_news", split="train")
        for r in tqdm(ds, desc="  FakeNews"):
            if len(rows) >= limit:
                break
            text = str(r.get("text", "")).strip()
            if len(text) < 20:
                continue
            is_fake = (int(r.get("label", 1)) == 0)
            intent  = 1 if is_fake else 0
            fact    = 0.1 if is_fake else 0.8  # proxy only (not regression target)
            # bias=-1, ekman=-1 — intent-only row
            rows.append(_row(text[:2000], -1, fact, intent, -1, "US", "fakenews"))
        print("  [OK] FakeNews: {:,} rows".format(len(rows)))
    except Exception as e:
        print("  [SKIP] FakeNews: {}".format(e))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 4: GoEmotions — emotion-only rows
# bias=-1  |  fact=proxy  |  intent=-1  |  emo=one-hot
# Neutral cap = 600 → ensures neutral < 20% of 3,000 emotion rows
# ─────────────────────────────────────────────────────────────────────────────
def load_emotions(limit=TASK_ROW_CAP, neutral_cap=600):
    """
    Unmapped GoEmotions labels are SKIPPED (not funnelled to Neutral).
    neutral_cap=600 ensures Neutral <= 20% of emotion rows at limit=3,000.
    """
    rows = []
    neutral_count = 0
    try:
        print("\n[4/4] GoEmotions (emotion-only rows, limit={:,}, neutral_cap={:,})...".format(
            limit, neutral_cap))
        try:
            ds = load_dataset("go_emotions", split="train")
        except Exception as e:
            print("  [WARN] Falling back to parquet: {}".format(e))
            ds = load_dataset(
                "parquet",
                data_files="hf://datasets/google-research-datasets/go_emotions/data/train-*",
                split="train"
            )

        for r in tqdm(ds, desc="  GoEmotions"):
            if len(rows) >= limit:
                break
            text = str(r.get("text", "")).strip()
            if len(text) < 10:
                continue
            go_labels = r.get("labels", [])
            if not go_labels:
                continue

            primary_go = int(go_labels[0])

            # Skip unmapped labels entirely — do NOT funnel to Neutral
            if primary_go not in GO_TO_EKMAN:
                continue
            ekman = GO_TO_EKMAN[primary_go]

            if ekman == 6:  # Neutral
                if neutral_count >= neutral_cap:
                    continue
                neutral_count += 1

            # bias=-1, intent=-1, fact=-1.0 (no fact label) — emotion-only row
            rows.append(_row(text, -1, -1.0, -1, ekman, "other", "goemo"))

        print("  [OK] GoEmotions: {:,} rows  neutral={:,} ({:.1f}%)".format(
            len(rows), neutral_count,
            neutral_count / max(len(rows), 1) * 100))
    except Exception as e:
        print("  [SKIP] GoEmotions: {}".format(e))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION ONLY — SemEval-2019 Task 4
# ─────────────────────────────────────────────────────────────────────────────
def save_semeval_validation(output_path="validation_intent.csv"):
    rows = []
    try:
        print("\n[VAL] SemEval-2019 Task 4 (intent validation only — never in training)...")
        ds = load_dataset(
            "hyperpartisan_news_detection", "byarticle",
            split="train", trust_remote_code=False
        )
        for r in tqdm(ds, desc="  SemEval-VAL"):
            text = str(r.get("text", r.get("title", ""))).strip()
            if len(text) < 15:
                continue
            intent = 1 if bool(r.get("hyperpartisan", False)) else 0
            rows.append({"text": text[:3000], "intent_label": intent,
                         "source": "semeval2019_byarticle"})
        df_val = pd.DataFrame(rows)
        df_val.to_csv(output_path, index=False)
        print("  [OK] SemEval validation: {:,} rows → {}".format(len(df_val), output_path))
    except Exception as e:
        print("  [SKIP] SemEval validation: {}".format(e))


# ─────────────────────────────────────────────────────────────────────────────
# Distribution Assertions
# ─────────────────────────────────────────────────────────────────────────────
def assert_distribution(df):
    total = len(df)
    print("\n" + "=" * 66)
    print("  FINAL DISTRIBUTION  ({:,} rows total)".format(total))
    print("=" * 66)
    errors = []

    # ── Bias (BABE rows only) ──────────────────────────────────────────────
    babe_df    = df[df["_source"] == "babe"] if "_source" in df.columns else df[df["bias_label"] >= 0]
    babe_total = len(babe_df)
    print("\n  BIAS  ({:,} BABE rows):".format(babe_total))
    for cls in range(3):
        cnt = len(babe_df[babe_df["bias_label"] == cls])
        pct = cnt / max(babe_total, 1) * 100
        print("    [{}] {:8s} {:>5,} ({:.1f}% of bias rows)".format(cls, BIAS_NAMES[cls], cnt, pct))
        if babe_total > 0 and not (25.0 <= pct <= 45.0):
            errors.append("Bias class {} out of balance: {:.1f}% (want 25-45%)".format(cls, pct))

    # Non-bias rows must all have bias_label == -1
    non_babe_with_bias = len(df[(df["_source"] != "babe") & (df["bias_label"] >= 0)]) if "_source" in df.columns else 0
    if non_babe_with_bias > 0:
        errors.append("Non-BABE rows with bias_label >= 0: {:,} (should be 0)".format(non_babe_with_bias))

    # ── Factuality ─────────────────────────────────────────────────────────
    liar_df = df[df["_source"] == "liar"] if "_source" in df.columns else df[df["fact_score"] >= 0]
    print("\n  FACTUALITY  ({:,} LIAR rows):".format(len(liar_df)))
    if len(liar_df) > 0:
        print("    min={:.2f}  max={:.2f}  mean={:.2f}".format(
            liar_df["fact_score"].min(), liar_df["fact_score"].max(), liar_df["fact_score"].mean()))
        if not (liar_df["fact_score"].min() >= 0.0 and liar_df["fact_score"].max() <= 1.0):
            errors.append("LIAR fact_score out of [0,1]")

    # Non-liar rows should have fact_score == -1.0 for sources that have no fact label
    # BABE and GoEmotions rows have no fact label → must be -1.0
    # FakeNews rows use proxy scores (0.1/0.8) — allowed since they encode real signal
    for src in ["babe", "goemo"]:
        if "_source" in df.columns:
            src_df = df[df["_source"] == src]
            non_sentinel = src_df[src_df["fact_score"] >= 0]
            if len(non_sentinel) > 0:
                errors.append("{} rows with fact_score >= 0: {:,} (should be 0)".format(
                    src, len(non_sentinel)))

    # ── Intent ─────────────────────────────────────────────────────────────
    fn_df = df[df["_source"] == "fakenews"] if "_source" in df.columns else df[df["intent_label"] >= 0]
    print("\n  INTENT  ({:,} FakeNews rows):".format(len(fn_df)))
    for lbl, name in {0: "News", 1: "Satire/Fake"}.items():
        cnt = len(fn_df[fn_df["intent_label"] == lbl])
        print("    [{}] {:12s} {:>5,}".format(lbl, name, cnt))

    # ── Emotion ────────────────────────────────────────────────────────────
    emo_df = df[df["_source"] == "goemo"] if "_source" in df.columns else df[df["_primary_ekman"] >= 0]
    emo_total = len(emo_df)
    print("\n  EMOTION  ({:,} GoEmotions rows):".format(emo_total))
    neutral_cnt = 0
    for cls in range(N_EKMAN):
        if "_primary_ekman" in df.columns:
            cnt = len(emo_df[emo_df["_primary_ekman"] == cls])
        else:
            cnt = 0
        if cls == 6:
            neutral_cnt = cnt
        pct = cnt / max(emo_total, 1) * 100
        print("    [{}] {:8s} {:>5,} ({:.1f}%)".format(cls, EKMAN_NAMES.get(cls, "?"), cnt, pct))
    neutral_pct = neutral_cnt / max(emo_total, 1) * 100
    if emo_total > 0 and neutral_pct >= 25.0:
        errors.append("Neutral emotion {:.1f}% >= 25% (want < 25%)".format(neutral_pct))

    # Confirm non-emotion rows have all-zeros emotion_label
    import json as _json
    def _is_sentinel_emo(s):
        try: return sum(_json.loads(s)) == 0.0
        except: return True
    if "_source" in df.columns:
        for src in ["babe", "liar", "fakenews"]:
            src_df = df[df["_source"] == src]
            bad = src_df[~src_df["emotion_label"].apply(_is_sentinel_emo)]
            if len(bad) > 0:
                errors.append("{} rows with non-zero emotion_label: {:,} (should be 0)".format(src, len(bad)))

    # ── Task row counts ────────────────────────────────────────────────────
    print("\n  TASK ROW COUNTS:")
    for src, label in [("babe","Bias"), ("liar","Factuality"),
                        ("fakenews","Intent"), ("goemo","Emotion")]:
        if "_source" in df.columns:
            cnt = len(df[df["_source"] == src])
            print("    {:14s} {:>6,}".format(label, cnt))

    print("=" * 66)
    if errors:
        print("\nFAILED CONSTRAINTS:")
        for e in errors: print("  * " + e)
        raise AssertionError("Distribution constraints failed.")
    else:
        print("\nAll constraints passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(output_path="training_data.csv", val_path="validation_intent.csv"):
    print("=" * 66)
    print("  CredibleX Dataset Builder v6")
    print("  US-Only | 3-Class Bias | Sentinel Labels | Task-Capped 3k")
    print("=" * 66)

    # Step 1: Bias rows (BABE, balanced)
    babe_buckets = load_babe()
    bias_rows    = balance_bias(babe_buckets)

    # Step 2: Task-specific rows (each capped at TASK_ROW_CAP=3,000)
    liar_rows    = load_liar()
    intent_rows  = load_fakenews()
    emotion_rows = load_emotions()

    # Step 3: Save SemEval as validation only
    save_semeval_validation(val_path)

    # Step 4: Combine and clean
    all_rows = bias_rows + liar_rows + intent_rows + emotion_rows
    print("\n  Raw total: {:,} rows".format(len(all_rows)))

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["text", "emotion_label"])
    df = df[df["text"].str.strip().str.len() >= 15].reset_index(drop=True)
    print("  After cleaning: {:,} rows".format(len(df)))

    # Step 5: Validate
    assert_distribution(df)

    # Step 6: Backup + Save (drop internal helper columns)
    if os.path.isfile(output_path):
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_path.replace(".csv", "_backup_{}.csv".format(ts))
        shutil.copy2(output_path, backup)
        print("\n  Backed up → {}".format(backup))

    drop_cols = [c for c in ["_primary_ekman", "_source"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print("  Saved {:,} rows → {}".format(len(df), output_path))
    return df


def check_existing(path="training_data.csv"):
    if not os.path.isfile(path):
        print("Not found: " + path); return
    df = pd.read_csv(path)
    print("Loaded {:,} rows".format(len(df)))
    print("Columns:", list(df.columns))
    print("NaN counts:\n", df.isnull().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true")
    parser.add_argument("--output", default="training_data.csv")
    parser.add_argument("--val",    default="validation_intent.csv")
    args = parser.parse_args()
    if args.check:
        check_existing(args.output)
    else:
        build_dataset(args.output, args.val)
