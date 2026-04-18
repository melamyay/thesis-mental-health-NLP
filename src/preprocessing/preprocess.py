"""
preprocessing/preprocess.py

Merges all raw Reddit forum XLSX files, cleans text data,
detects language, and exports a clean CSV ready for NLP analysis.

Usage:
    python src/preprocessing/preprocess.py \
        --input_dir data/raw/forums_reddit \
        --output data/processed/reddit_clean.csv
"""

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RAW_FILES = [
    "posts_of_college_new.xlsx",
    "posts_of_collegerant_new.xlsx",
    "posts_of_csMajors_new.xlsx",
    "posts_of_etudiants_new.xlsx",
    "posts_of_Student_new.xlsx",
    "posts_of_Students_new.xlsx",
]

FORUM_LANG = {
    "etudiants": "fr",
    "college":   "en",
    "CollegeRant": "en",
    "csMajors":  "en",
    "Student":   "en",
    "Students":  "en",
}

# Columns to keep in the final output
KEEP_COLS = [
    "id_post",
    "forum",
    "auteur",
    "titre",
    "texte",
    "texte_clean",
    "texte_source",   # titre | texte | titre+texte
    "categorie",
    "nb_commentaires",
    "nb_reactions",
    "date_heure_post",
    "annee_post",
    "mois_post",
    "langue",
    "longueur_texte_clean",
    "is_empty",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text)

def _remove_reddit_artifacts(text: str) -> str:
    # Remove u/username and r/subreddit mentions
    text = re.sub(r"\bu/\w+", "", text)
    text = re.sub(r"\br/\w+", "", text)
    return text

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single text string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _normalize_unicode(text)
    text = _remove_urls(text)
    text = _remove_reddit_artifacts(text)
    text = _normalize_whitespace(text)
    return text

# ── Core pipeline ─────────────────────────────────────────────────────────────

def load_and_merge(input_dir: Path) -> pd.DataFrame:
    """Load all XLSX files and merge into a single DataFrame."""
    dfs = []
    for filename in RAW_FILES:
        path = input_dir / filename
        # Support filenames with numeric prefix (e.g. 177649_posts_of_college_new.xlsx)
        if not path.exists():
            matches = list(input_dir.glob(f"*{filename}"))
            if not matches:
                log.warning(f"File not found, skipping: {filename}")
                continue
            path = matches[0]
        df = pd.read_excel(path)
        log.info(f"Loaded {path.name}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    log.info(f"Total rows after merge: {len(merged)}")
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and derive new columns."""

    # 1. Drop exact duplicate posts
    before = len(df)
    df = df.drop_duplicates(subset="id_post", keep="first")
    log.info(f"Removed {before - len(df)} duplicate posts")

    # 2. Detect language from forum name
    df["langue"] = df["forum"].map(FORUM_LANG).fillna("unknown")

    # 3. Clean text
    df["texte_clean"] = df["texte"].apply(
        lambda x: clean_text(x) if pd.notna(x) else ""
    )

    # 4. Flag empty texts (NaN or URL-only posts)
    df["is_empty"] = df["texte_clean"] == ""

    # 5. For empty texts, fall back to titre as text source
    df["texte_source"] = "texte"
    titre_fallback = df["is_empty"] & df["titre"].notna()
    df.loc[titre_fallback, "texte_clean"] = df.loc[titre_fallback, "titre"].apply(clean_text)
    df.loc[titre_fallback, "texte_source"] = "titre"

    # If both texte and titre are available, note it
    both_available = ~df["is_empty"] & df["titre"].notna()
    df.loc[both_available, "texte_source"] = "titre+texte"

    # 6. Text length after cleaning
    df["longueur_texte_clean"] = df["texte_clean"].str.len()

    # 7. Normalize categorie (strip whitespace)
    df["categorie"] = df["categorie"].str.strip()

    # 8. Parse date column
    df["date_heure_post"] = pd.to_datetime(df["date_heure_post"], errors="coerce")

    log.info(f"Empty texts after fallback: {(df['texte_clean'] == '').sum()}")
    log.info(f"Language distribution:\n{df['langue'].value_counts().to_string()}")
    log.info(f"Forum distribution:\n{df['forum'].value_counts().to_string()}")

    return df


def export(df: pd.DataFrame, output_path: Path) -> None:
    """Keep only relevant columns and export to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_cols = [c for c in KEEP_COLS if c in df.columns]
    df[final_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info(f"Exported {len(df)} rows → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess Reddit forum XLSX files")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw/forums_reddit"),
        help="Directory containing raw XLSX files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reddit_clean.csv"),
        help="Output CSV path"
    )
    args = parser.parse_args()

    log.info("=== Starting preprocessing pipeline ===")
    df = load_and_merge(args.input_dir)
    df = clean(df)
    export(df, args.output)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()