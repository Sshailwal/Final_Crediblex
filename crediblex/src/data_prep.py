"""
Data preparation pipeline for the Indian political bias dataset (CredibleX v2).

Tasks:
    - Text cleaning: strip URLs, preserve Devanagari, normalise whitespace, cap length.
    - Column normalisation: handles variant column names (text/content, label/bias).
    - Label validation: only classes {0, 1, 2, 3, 4} are kept.
    - Stratified 70 / 15 / 15 train / val / test split.
    - Saves artefacts to data/train.csv, data/val.csv, data/test.csv.

Label schema:
    0 = Far Left | 1 = Left | 2 = Center | 3 = Right | 4 = Far Right
"""

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Regex to strip URLs (http/https/ftp schemes plus bare www.)
_URL_RE = re.compile(
    r"https?://\S+|ftp://\S+|www\.\S+",
    re.IGNORECASE,
)
# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_VALID_LABELS: set[int] = {0, 1, 2, 3, 4}
_LABEL_NAMES: list[str] = ["Far Left", "Left", "Center", "Right", "Far Right"]
_MIN_TEXT_LEN: int = 30
_MAX_TEXT_LEN: int = 512


def clean_text(text: str) -> str:
    """
    Clean a raw news text string for model input.

    Steps:
        1. Remove URLs (http/https/ftp/www).
        2. Preserve Devanagari characters (U+0900–U+097F) — other Unicode kept too.
        3. Collapse multiple whitespace characters into a single space.
        4. Strip leading/trailing whitespace.
        5. Truncate to ``_MAX_TEXT_LEN`` characters.

    Args:
        text: Raw input string (may contain HTML artefacts, URLs, etc.).

    Returns:
        Cleaned, whitespace-normalised, length-capped string.
    """
    if not isinstance(text, str):
        text = str(text)
    # 1. Remove URLs
    text = _URL_RE.sub(" ", text)
    # 2. Normalise whitespace (keep Devanagari and all Unicode naturally)
    text = re.sub(r"[ \t]+", " ", text)     # multiple spaces/tabs → one space
    text = re.sub(r"\n+", " ", text)         # newlines → space
    text = text.strip()
    # 3. Cap at _MAX_TEXT_LEN characters (word-boundary-aware truncation would be
    #    nicer but the spec requires a hard character cap)
    return text[:_MAX_TEXT_LEN]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise variant column names to a canonical schema.

    Accepted variants:
        - text column: ``text``, ``content``, ``article``, ``headline``
        - label column: ``label``, ``bias``, ``bias_label``, ``class``

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        DataFrame with ``text`` and ``label`` columns guaranteed present.

    Raises:
        KeyError: If neither a text-like nor a label-like column can be found.
    """
    col_lower = {c.lower().strip(): c for c in df.columns}

    # ── text column ──────────────────────────────────────────────────────────
    text_candidates = ["text", "content", "article", "headline"]
    text_col: Optional[str] = None
    for cand in text_candidates:
        if cand in col_lower:
            text_col = col_lower[cand]
            break
    if text_col is None:
        raise KeyError(
            f"No recognisable text column found. Columns present: {list(df.columns)}"
        )

    # ── label column ─────────────────────────────────────────────────────────
    label_candidates = ["label", "bias", "bias_label", "class"]
    label_col: Optional[str] = None
    for cand in label_candidates:
        if cand in col_lower:
            label_col = col_lower[cand]
            break
    if label_col is None:
        raise KeyError(
            f"No recognisable label column found. Columns present: {list(df.columns)}"
        )

    df = df.rename(columns={text_col: "text", label_col: "label"})
    # Keep only the canonical columns (plus any extras that may be useful)
    return df


def prepare_splits(
    csv_path: str | Path,
    out_dir: str | Path = "data",
) -> None:
    """
    Load, clean, validate, and split the Indian bias dataset.

    Pipeline:
        1. Load CSV from *csv_path*.
        2. Normalise column names via :func:`_normalise_columns`.
        3. Apply :func:`clean_text` to the ``text`` column.
        4. Drop rows with null text/label.
        5. Cast labels to ``int``; drop rows whose label ∉ {0,1,2,3,4}.
        6. Drop rows where ``len(text) < 30`` after cleaning.
        7. Stratified 70 / 15 / 15 split (``random_state=42``).
        8. Save to *out_dir*/train.csv, val.csv, test.csv.
        9. Print class distribution for each split.

    Args:
        csv_path: Path to the raw labeled CSV file.
        out_dir:  Directory to write train/val/test CSVs into.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        KeyError:          If required columns are missing after normalisation.
        ValueError:        If the dataset is too small to split.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    logger.info("Loading dataset from %s …", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ── 1. Normalise columns ─────────────────────────────────────────────────
    df = _normalise_columns(df)

    # ── 2. Clean text ────────────────────────────────────────────────────────
    df["text"] = df["text"].apply(clean_text)

    # ── 3. Drop nulls ────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    logger.info("Dropped %d rows with null text/label.", before - len(df))

    # ── 4. Validate labels ───────────────────────────────────────────────────
    try:
        df["label"] = df["label"].astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not cast label column to int: {exc}") from exc

    before = len(df)
    df = df[df["label"].isin(_VALID_LABELS)]
    logger.info("Dropped %d rows with invalid labels.", before - len(df))

    # ── 5. Remove short texts ────────────────────────────────────────────────
    before = len(df)
    df = df[df["text"].str.len() >= _MIN_TEXT_LEN]
    logger.info("Dropped %d rows shorter than %d chars.", before - len(df), _MIN_TEXT_LEN)

    if len(df) < 10:
        raise ValueError(
            f"Dataset too small after filtering ({len(df)} rows). "
            "Check your CSV for data quality issues."
        )

    df = df.reset_index(drop=True)
    logger.info("Final usable rows: %d", len(df))

    # ── 6. Stratified splits ─────────────────────────────────────────────────
    # 70 train / 30 temp → 50% of 30 = 15 val, 15 test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    # ── 7. Save ──────────────────────────────────────────────────────────────
    train_path = out_dir / "train.csv"
    val_path   = out_dir / "val.csv"
    test_path  = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved: %s (%d rows)", train_path, len(train_df))
    logger.info("Saved: %s (%d rows)", val_path,   len(val_df))
    logger.info("Saved: %s (%d rows)", test_path,  len(test_df))

    # ── 8. Print class distribution ──────────────────────────────────────────
    def _distribution(split_df: pd.DataFrame, name: str) -> None:
        print(f"\n{'--' * 25}")
        print(f"  {name} split -- {len(split_df)} samples")
        print(f"{'--' * 25}")
        counts = split_df["label"].value_counts().sort_index()
        for label_id, count in counts.items():
            label_name = _LABEL_NAMES[int(label_id)] if int(label_id) < len(_LABEL_NAMES) else "?"
            pct = count / len(split_df) * 100
            bar = "#" * int(pct / 2)
            print(f"  [{label_id}] {label_name:<10}  {count:>5} ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 50)
    print("  CLASS DISTRIBUTION AFTER SPLITTING")
    print("=" * 50)
    _distribution(train_df, "TRAIN")
    _distribution(val_df,   "VAL  ")
    _distribution(test_df,  "TEST ")
    print()


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/indian_bias_labeled.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data"
    prepare_splits(csv, out)
