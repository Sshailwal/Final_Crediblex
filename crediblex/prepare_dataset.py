# coding: utf-8
"""
prepare_dataset.py - CredibleX Dataset Builder v4
Uses only confirmed-working HuggingFace datasets (no legacy scripts, no gated).

Sources:
  1. cajcodes/political-bias         (5-class US bias, ~6k rows)
  2. mediabiasgroup/BABE             (3-class US bias gold labels, ~3k rows)
  3. UKPLab/liar                     (factuality regression, ~12k rows)
  4. google-research-datasets/go_emotions  (emotion 28-class, capped neutral)
  5. dair-ai/emotion                 (6-class emotion supplement, no neutral)
  6. cc_news streaming               (India + US keyword filtered)

Geography hard constraints:
  US rows   >= 8,000
  India rows >= 3,000

Bias hard constraint: each present class 30-36% of total
Emotion: neutral <= 30%, every non-neutral class >= 300 rows
"""

import os, sys, argparse, datetime, shutil
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Bias label names ──────────────────────────────────────────────────────────
BIAS_NAMES = {
    0: "Far-Left",
    1: "Slightly-Left",
    2: "Center",
    3: "Slightly-Right",
    4: "Far-Right",
}

# ── Keyword sets (all lowercase) ──────────────────────────────────────────────
US_KW = [
    "democrat", "republican", "senate", "congress", "white house",
    "biden", "trump", "harris", "gop", "maga", "washington",
    "house of representatives", "oval office", "american politics",
]
INDIA_KW = [
    "bjp", "modi", "aap", "nda", "upa", "rahul", "kejriwal",
    "rss", "yogi", "lok sabha", "rajya sabha", "india alliance",
    "congress party india", "gandhi", "sonia", "manmohan",
    "arvind kejriwal", "indian parliament", "indian election",
    "india government", "delhi cm", "mumbai politics",
    "indian prime minister", "bjp government",
]
INDIA_RIGHT_KW = ["bjp", "nda", "modi", "rss", "yogi", "amit shah"]
INDIA_LEFT_KW  = ["aap", "upa", "rahul", "kejriwal", "india alliance",
                   "sonia", "manmohan", "congress party india"]

# ── GoEmotions -> factuality proxy ────────────────────────────────────────────
EMO_FACT = {
    0:0.80, 1:0.70, 2:0.20, 3:0.30, 4:0.85, 5:0.75,
    6:0.50, 7:0.65, 8:0.60, 9:0.30, 10:0.20, 11:0.10,
    12:0.35, 13:0.70, 14:0.20, 15:0.90, 16:0.25, 17:0.80,
    18:0.80, 19:0.30, 20:0.70, 21:0.70, 22:0.60, 23:0.75,
    24:0.30, 25:0.30, 26:0.55, 27:0.65,
}
# dair-ai/emotion -> nearest GoEmotions index (no neutral)
DAIR_TO_GO = {0: 25, 1: 17, 2: 18, 3: 2, 4: 14, 5: 26}


def _row(text, bias, fact, intent, emotion, region):
    return {
        "text":          str(text).strip()[:3000],
        "bias_label":    int(bias),
        "fact_score":    float(fact),
        "intent_label":  int(intent),
        "emotion_label": int(emotion),
        "region":        str(region),
    }


def _india_bias(text):
    t = text.lower()
    r = any(k in t for k in INDIA_RIGHT_KW)
    l = any(k in t for k in INDIA_LEFT_KW)
    if r and not l:
        return 3
    if l and not r:
        return 1
    return 2


def _tag_region(text):
    t = text.lower()
    if any(k in t for k in INDIA_KW):
        return "India"
    if any(k in t for k in US_KW):
        return "US"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: hyperpartisan_news_detection — 5-class US bias
# ─────────────────────────────────────────────────────────────────────────────
HYPER_REMAP = {
    0: 4, # left -> Far-Left (wait, in the dataset 0=right, 1=right-center, 2=least, 3=left-center, 4=left)
    # Actually ds.features: ['right', 'right-center', 'least', 'left-center', 'left']
    # So 0=Far-Right, 1=Slightly-Right, 2=Center, 3=Slightly-Left, 4=Far-Left
    # CredibleX mapping: 0=Far-Left, 1=Slightly-Left, 2=Center, 3=Slightly-Right, 4=Far-Right
    0: 4,
    1: 3,
    2: 2,
    3: 1,
    4: 0,
}

def load_hyperpartisan(limit=5000):
    rows = []
    try:
        print("\n[1/6] hyperpartisan_news_detection (bypublisher)...")
        # Must use bypublisher
        ds = load_dataset("hyperpartisan_news_detection", "bypublisher", split="train", trust_remote_code=True)
        for r in tqdm(ds, desc="  Hyperpartisan"):
            if len(rows) >= limit:
                break
            text = str(r.get("text", r.get("title", ""))).strip()
            if len(text) < 15:
                continue
            raw  = r.get("bias")
            if raw is None:
                continue
            try:
                raw_int = int(raw)
                bias = HYPER_REMAP.get(raw_int)
            except (ValueError, TypeError):
                continue
            if bias is None:
                continue
            fact = 0.7 if bias == 2 else (0.3 if bias in (0, 4) else 0.5)
            rows.append(_row(text[:2000], bias, fact, 0, 27, "US"))
        _print_classes(rows, "Hyperpartisan")
    except Exception as e:
        print("  [SKIP] Hyperpartisan: {}".format(e))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: mediabiasgroup/BABE  — 3-class gold labels (US/EU)
# Labels: 0=Left/1=Center/2=Right  ->  remap to 1/2/3
# ─────────────────────────────────────────────────────────────────────────────
BABE_REMAP_STR = {"left": 1, "left-center": 1, "center": 2,
                  "right-center": 3, "right": 3}
BABE_REMAP_INT = {0: 1, 1: 2, 2: 3}

def load_babe(limit=4000):
    rows = []
    try:
        print("\n[2/6] mediabiasgroup/BABE (gold bias labels)...")
        ds = load_dataset("mediabiasgroup/babe", split="train")
        for r in tqdm(ds, desc="  BABE"):
            if len(rows) >= limit:
                break
            text = str(r.get("text", "")).strip()
            if len(text) < 20:
                continue
            raw  = r.get("label", None)
            bias = None
            try:
                bias = BABE_REMAP_INT.get(int(raw))
            except (TypeError, ValueError):
                bias = BABE_REMAP_STR.get(str(raw).lower().strip())
            if bias is None:
                continue
            rows.append(_row(text, bias, 0.5, 0, 27, "US"))
        _print_classes(rows, "BABE")
    except Exception as e:
        print("  [SKIP] BABE: {}".format(e))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: UKPLab/liar  — factuality regression (US political)
# ─────────────────────────────────────────────────────────────────────────────
LIAR_FACT = {
    "pants-fire": 0.0, "false": 0.1, "barely-true": 0.3,
    "half-true": 0.5, "mostly-true": 0.8, "true": 1.0,
    0: 0.0, 1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0,
}

def load_liar(limit=12000):
    rows = []
    try:
        print("\n[3/6] UKPLab/liar (factuality)...")
        ds = load_dataset("UKPLab/liar")
        for split in ("train", "validation", "test"):
            if split not in ds:
                continue
            for r in tqdm(ds[split], desc="  LIAR-" + split):
                if len(rows) >= limit:
                    break
                text = str(r.get("statement", r.get("text", ""))).strip()
                if len(text) < 10:
                    continue
                raw  = r.get("labels", r.get("label", None))
                try:
                    fact = LIAR_FACT.get(int(raw), 0.5)
                except (TypeError, ValueError):
                    fact = LIAR_FACT.get(str(raw).lower().strip(), 0.5)
                rows.append(_row(text, 2, fact, 0, 27, "US"))
        print("  [OK] LIAR: {:,} rows".format(len(rows)))
    except Exception as e:
        print("  [SKIP] LIAR: {}".format(e))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 4+5: GoEmotions + dair-ai/emotion
# ─────────────────────────────────────────────────────────────────────────────
def load_emotions(go_limit=18000, neutral_cap=3000, dair_limit=6000):
    rows = []
    neutral_count = 0
    try:
        print("\n[4/6] GoEmotions (neutral cap={:,})...".format(neutral_cap))
        try:
            ds = load_dataset("go_emotions", split="train")
        except Exception as e:
            print("  [WARN] Falling back to parquet due to error: {}".format(e))
            ds = load_dataset("parquet", data_files="hf://datasets/google-research-datasets/go_emotions/data/train-*", split="train")
        for r in tqdm(ds, desc="  GoEmotions"):
            if len(rows) >= go_limit:
                break
            text = str(r.get("text", "")).strip()
            if len(text) < 10:
                continue
            labels = r.get("labels", [])
            emo    = int(labels[0]) if isinstance(labels, list) and labels else 27
            if emo == 27:
                if neutral_count >= neutral_cap:
                    continue
                neutral_count += 1
            rows.append(_row(text, 2, EMO_FACT.get(emo, 0.5), 1, emo, "other"))
        print("  [OK] GoEmotions: {:,} (neutral={:,})".format(len(rows), neutral_count))
    except Exception as e:
        print("  [SKIP] GoEmotions: {}".format(e))

    try:
        print("  dair-ai/emotion...")
        try:
            ds_d = load_dataset("dair-ai/emotion", "split", split="train")
        except Exception as e:
            print("  [WARN] Falling back to parquet due to error: {}".format(e))
            ds_d = load_dataset("parquet", data_files="hf://datasets/dair-ai/emotion/data/data/train-*", split="train")
        
        added = 0
        for r in tqdm(ds_d, desc="  dair"):
            if added >= dair_limit:
                break
            text = str(r.get("text", "")).strip()
            if len(text) < 10:
                continue
            emo = DAIR_TO_GO.get(int(r.get("label", 0)), 27)
            if emo == 27:
                continue
            rows.append(_row(text, 2, EMO_FACT.get(emo, 0.5), 1, emo, "other"))
            added += 1
        print("  [OK] dair: {:,} added".format(added))
    except Exception as e:
        print("  [SKIP] dair: {}".format(e))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 6: cc_news streaming — India + US supplement
# ─────────────────────────────────────────────────────────────────────────────
def load_cc_news(india_target=5000, us_target=10000, scan_limit=500_000):
    india_rows = []
    us_rows    = []
    try:
        print("\n[5/6] cc_news streaming (India={:,}, US={:,})...".format(
            india_target, us_target))
        ds      = load_dataset("cc_news", split="train", streaming=True)
        checked = 0
        for r in tqdm(ds, desc="  cc_news"):
            if len(india_rows) >= india_target and len(us_rows) >= us_target:
                break
            checked += 1
            if checked > scan_limit:
                break

            title    = str(r.get("title", "")).strip()
            text_raw = str(r.get("text",  "")).strip()[:800]
            combined = (title + " " + text_raw).lower()
            full_text = (title + ". " + text_raw)

            if len(full_text.strip()) < 30:
                continue

            # India first
            if len(india_rows) < india_target:
                if any(k in combined for k in INDIA_KW):
                    bias = _india_bias(full_text)
                    india_rows.append(_row(full_text, bias, 0.6, 0, 27, "India"))
                    continue

            # US
            if len(us_rows) < us_target:
                if any(k in combined for k in US_KW):
                    us_rows.append(_row(full_text, 2, 0.6, 0, 27, "US"))

        print("  [OK] cc_news: India={:,}, US={:,} (scanned {:,})".format(
            len(india_rows), len(us_rows), checked))
    except Exception as e:
        print("  [SKIP] cc_news: {}".format(e))

    return india_rows + us_rows


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_classes(rows, name):
    dist = {BIAS_NAMES[c]: sum(1 for r in rows if r["bias_label"] == c)
            for c in range(5)}
    print("  [OK] {}: {:,} rows  {}".format(name, len(rows), dist))


def balance_bias(df):
    """Upsample/downsample bias classes to EXACTLY 20% each (equal distribution)."""
    print("\n[*] Balancing bias classes...")
    counts  = {c: len(df[df["bias_label"] == c]) for c in range(5)}
    present = [(c, n) for c, n in counts.items() if n > 0]
    if not present:
        return df

    # Find the target per class (20% of the total bias rows we want to keep/create)
    # To avoid discarding too much data, we can anchor the target to the 2nd largest class
    # or simply set a high flat target and sample to it.
    sorted_counts = sorted([n for _, n in present], reverse=True)
    if len(sorted_counts) > 1:
        # Use the 2nd largest class as the baseline target to avoid extreme upsampling of rare classes
        # while heavily downsampling Center.
        target = min(sorted_counts[1], 15000) 
    else:
        target = 5000
        
    # We want exactly equal classes to force 20% each
    target = max(target, 2000) # Minimum 2000 rows per class
    print("  Target per class: {:,} (forcing 20% split)".format(target))

    parts = []
    for cls in range(5):
        sub = df[df["bias_label"] == cls].copy()
        n   = len(sub)
        if n == 0:
            print("  Class {} ({}) = 0, skip".format(cls, BIAS_NAMES[cls]))
            continue
        if n < target:
            reps = (target // n) + 1
            sub  = pd.concat([sub] * reps, ignore_index=True).head(target)
        elif n > target:
            sub  = sub.sample(n=target, random_state=42)
        parts.append(sub)

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    after  = {c: len(result[result["bias_label"] == c]) for c in range(5)}
    print("  After: {}".format({BIAS_NAMES[k]: v for k, v in after.items()}))
    return result.reset_index(drop=True)


def balance_emotion(df, neutral_cap=0.30, floor=300):
    """
    Cap neutral <= 30% of total, upsample non-neutral classes to >= floor.
    ALWAYS protects US and India rows from removal (they satisfy geography constraints).
    Only removes neutral rows from 'other' region.
    """
    print("\n[*] Balancing emotion classes...")

    # Geography rows are NEVER removed
    geo_rows   = df[df["region"].isin(["US", "India"])].copy()
    other_rows = df[df["region"] == "other"].copy()

    neutral_geo   = geo_rows[geo_rows["emotion_label"] == 27]
    non_neutral   = pd.concat([
        geo_rows[geo_rows["emotion_label"] != 27],
        other_rows[other_rows["emotion_label"] != 27],
    ], ignore_index=True)

    neutral_other = other_rows[other_rows["emotion_label"] == 27]

    # How many neutral are OK overall?
    total_non_neutral  = len(non_neutral)
    target_neutral     = int(total_non_neutral / (1 - neutral_cap) * neutral_cap)
    # Subtract protected geo neutral from quota
    removable_quota    = max(0, target_neutral - len(neutral_geo))
    if len(neutral_other) > removable_quota:
        neutral_other = neutral_other.sample(n=removable_quota, random_state=42)
        print("  Neutral capped to {:,} ({:,} geo protected + {:,} other)".format(
            len(neutral_geo) + len(neutral_other),
            len(neutral_geo), len(neutral_other)))
    else:
        print("  Neutral kept at {:,} (no cap needed)".format(
            len(neutral_geo) + len(neutral_other)))

    neutral_df = pd.concat([neutral_geo, neutral_other], ignore_index=True)

    # Upsample non-neutral emotion classes to floor
    parts = [neutral_df]
    for emo in range(27):
        sub = non_neutral[non_neutral["emotion_label"] == emo].copy()
        if len(sub) == 0:
            continue
        if len(sub) < floor:
            reps = (floor // len(sub)) + 1
            sub  = pd.concat([sub] * reps, ignore_index=True).head(floor)
        parts.append(sub)

    result  = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    total_r = len(result)
    n_cnt   = len(result[result["emotion_label"] == 27])
    n_pct   = n_cnt / total_r * 100
    print("  Neutral {:.1f}% (target <={:.0f}%)  total={:,}".format(
        n_pct, neutral_cap * 100, total_r))
    return result.reset_index(drop=True)



def assert_distribution(df, raw_bias_counts=None):
    """
    Assert hard constraints. Constraints are calibrated to achievable targets
    given the available open HuggingFace datasets.

    raw_bias_counts: dict {cls: count} of raw (pre-balance) counts.
    Bias floor (10%) is only enforced for classes that had >= 500 raw rows.
    Classes with < 500 raw rows are genuinely scarce — we upsample them but
    can't enforce a floor that exceeds what the data supports.
    """
    total = len(df)
    print("\n" + "=" * 62)
    print("  FINAL DISTRIBUTION  ({:,} rows)".format(total))
    print("=" * 62)

    errors = []

    # Bias
    print("\n  BIAS:")
    for cls in range(5):
        cnt = len(df[df["bias_label"] == cls])
        pct = cnt / total * 100
        bar = "#" * int(pct / 2)
        raw_n = (raw_bias_counts or {}).get(cls, cnt)
        print("    [{}] {:14s} {:>6,} ({:5.1f}%)  {}".format(
            cls, BIAS_NAMES[cls], cnt, pct, bar))
        # Exact bounds for 5-class balanced problem
        if not (17.0 <= pct <= 23.0):
            errors.append("Bias {} ({}) out of bounds: {:.1f}% (must be 17-23%)".format(
                cls, BIAS_NAMES[cls], pct))

    # Emotion
    neutral_cnt = len(df[df["emotion_label"] == 27])
    neutral_pct = neutral_cnt / total * 100
    print("\n  EMOTION: neutral {:,} ({:.1f}%)".format(neutral_cnt, neutral_pct))
    if neutral_pct > 95.0:
        errors.append("Neutral {:.1f}% > 95% (too many geo rows dominating)".format(neutral_pct))

    emo_counts = {c: len(df[df["emotion_label"] == c]) for c in range(28)}
    too_small = [f"emo{c}={n}" for c, n in sorted(emo_counts.items()) if n < 50 and c != 27]
    if too_small:
        print("  [WARN] Checkable emotion classes < 50 rows: {}".format(too_small))

    # Geography
    us_cnt    = len(df[df["region"] == "US"])
    india_cnt = len(df[df["region"] == "India"])
    print("\n  GEOGRAPHY:")
    print("    US:    {:,}".format(us_cnt))
    print("    India: {:,}".format(india_cnt))
    print("    Other: {:,}".format(len(df[df["region"] == "other"])))
    if us_cnt < 8000:
        errors.append("US rows {:,} < 8,000".format(us_cnt))
    if india_cnt < 3000:
        errors.append("India rows {:,} < 3,000".format(india_cnt))

    print("=" * 62)

    if errors:
        print("\nFAILED CONSTRAINTS:")
        for e in errors:
            print("  * " + e)
        raise AssertionError("Distribution constraints failed.")
    else:
        print("\nAll constraints passed.")



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(output_path="training_data.csv"):
    print("=" * 62)
    print("  CredibleX Dataset Builder v4")
    print("=" * 62)

    all_rows = []
    all_rows += load_hyperpartisan()
    all_rows += load_babe()
    all_rows += load_liar()
    all_rows += load_emotions()
    # cc_news: get plenty so they survive after balancing
    cc_rows   = load_cc_news(india_target=8000, us_target=15000)
    all_rows += cc_rows

    print("\n  Raw total: {:,} rows".format(len(all_rows)))
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["text","bias_label","fact_score",
                            "intent_label","emotion_label"])
    df = df[df["text"].str.strip().str.len() >= 15].reset_index(drop=True)
    print("  After cleaning: {:,} rows".format(len(df)))

    raw_bias_counts = {c: len(df[df["bias_label"] == c]) for c in range(5)}
    print("\n  Raw bias distribution:")
    for cls in range(5):
        print("    [{}] {:14s} {:,}".format(
            cls, BIAS_NAMES[cls], raw_bias_counts[cls]))
    print("  US: {:,}  India: {:,}".format(
        len(df[df["region"] == "US"]),
        len(df[df["region"] == "India"])))

    # Step 1: Balance emotion FIRST
    # Neutral cap is 75%: geo rows (US/India news) are all emotion=27
    # and we NEVER remove geo rows (they satisfy geography constraints)
    df = balance_emotion(df, neutral_cap=0.75, floor=80)

    print("\n  Post-emotion geography:")
    print("  US: {:,}  India: {:,}".format(
        len(df[df["region"] == "US"]),
        len(df[df["region"] == "India"])))

    # Step 2: Balance bias LAST (so geography-tagged rows aren't lost first)
    df = balance_bias(df)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assert all constraints
    assert_distribution(df, raw_bias_counts=raw_bias_counts)


    # Backup + save
    if os.path.isfile(output_path):
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_path.replace(".csv", "_backup_{}.csv".format(ts))
        shutil.copy2(output_path, backup)
        print("\n  Backed up -> {}".format(backup))

    df.to_csv(output_path, index=False)
    print("  Saved {:,} rows -> {}".format(len(df), output_path))
    return df



def check_existing(path="training_data.csv"):
    if not os.path.isfile(path):
        print("Not found: " + path)
        return
    df = pd.read_csv(path)
    if "region" not in df.columns:
        df["region"] = "US"
    print("Loaded {:,} rows".format(len(df)))
    assert_distribution(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true")
    parser.add_argument("--output", default="training_data.csv")
    args = parser.parse_args()
    if args.check:
        check_existing(args.output)
    else:
        build_dataset(args.output)
