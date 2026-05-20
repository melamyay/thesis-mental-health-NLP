import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SAISON_MAP = {
    1:  "hiver",
    2:  "hiver",
    3:  "printemps",
    4:  "printemps",
    5:  "examens",
    6:  "ete",
    7:  "ete",
    8:  "ete",
    9:  "rentree",
    10: "automne",
    11: "automne",
    12: "examens",
}

SAISON_ORDER = ["rentree", "automne", "hiver", "printemps", "examens", "ete"]

EMOTION_COLS = [
    "emotion_anger", "emotion_anticipation", "emotion_disgust",
    "emotion_fear", "emotion_joy", "emotion_sadness",
    "emotion_surprise", "emotion_trust"
]

SENTIMENT_COLS = ["score_pos", "score_neu", "score_neg", "score_cont"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_mean(s): return round(float(s.mean()), 4) if len(s) > 0 else None
def safe_std(s):  return round(float(s.std()),  4) if len(s) > 1 else None
def dominant(s):  return s.value_counts().index[0] if len(s) > 0 else None

def aggregate_group(group: pd.DataFrame, labels: dict) -> dict:
    row = dict(labels)
    row["nb_posts"]      = len(group)
    row["nb_posts_en"]   = int((group["langue"] == "en").sum())
    row["nb_posts_fr"]   = int((group["langue"] == "fr").sum())

    # Sentiment
    for col in SENTIMENT_COLS:
        if col in group.columns:
            s = group[col].dropna()
            row[f"mean_{col}"] = safe_mean(s)
            row[f"std_{col}"]  = safe_std(s)

    if "sentiment_label" in group.columns:
        for label in ["positive", "neutral", "negative"]:
            row[f"pct_{label}"] = round((group["sentiment_label"] == label).sum() / len(group) * 100, 2)

    # Émotions
    for col in EMOTION_COLS:
        if col in group.columns:
            row[f"mean_{col.replace('emotion_', '')}"] = safe_mean(group[col].dropna())

    if "emotion_dominant" in group.columns:
        row["dominant_emotion"] = dominant(group["emotion_dominant"].dropna())

    # Distress
    if "distress_score" in group.columns:
        s = group["distress_score"].dropna()
        row["mean_distress"]      = safe_mean(s)
        row["std_distress"]       = safe_std(s)
        row["max_distress"]       = round(float(s.max()), 4) if len(s) > 0 else None
        row["pct_high_distress"]  = round((s >= 0.4).sum() / len(group) * 100, 2)

    if "distress_level" in group.columns:
        row["dominant_distress_level"] = dominant(group["distress_level"].dropna())

    # Topics
    if "topic_label" in group.columns:
        row["dominant_topic"]    = dominant(group["topic_label"].dropna())
        row["nb_unique_topics"]  = int(group["topic_label"].nunique())

    # Longueur
    if "nb_tokens" in group.columns:
        row["mean_nb_tokens"] = safe_mean(group["nb_tokens"].dropna())

    return row


# ── Aggregation functions ─────────────────────────────────────────────────────

def aggregate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating by month...")
    rows = []
    for (mois, langue, forum), group in df.groupby(
        ["mois_post", "langue", "forum"], dropna=True
    ):
        rows.append(aggregate_group(group, {
            "mois":    int(mois),
            "langue":  langue,
            "forum":   forum,
            "periode": f"{int(mois):02d}",
        }))
    result = pd.DataFrame(rows).sort_values(["mois", "langue", "forum"]).reset_index(drop=True)
    log.info(f"Monthly: {len(result):,} rows")
    return result


def aggregate_by_saison(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating by academic season...")
    df = df.copy()
    df["saison"] = df["mois_post"].map(SAISON_MAP)
    rows = []
    for (saison, langue, forum), group in df.groupby(
        ["saison", "langue", "forum"], dropna=True
    ):
        rows.append(aggregate_group(group, {
            "saison": saison,
            "langue": langue,
            "forum":  forum,
        }))
    result = pd.DataFrame(rows)
    result["saison"] = pd.Categorical(result["saison"], categories=SAISON_ORDER, ordered=True)
    result = result.sort_values(["saison", "langue", "forum"]).reset_index(drop=True)
    log.info(f"Seasonal: {len(result):,} rows")
    return result


def aggregate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating by day of week...")
    df = df.copy()
    df["date_heure_post"] = pd.to_datetime(df["date_heure_post"], errors="coerce")
    df["jour_semaine"]    = df["date_heure_post"].dt.day_name()
    df["jour_num"]        = df["date_heure_post"].dt.dayofweek
    rows = []
    for (jour, jour_num, langue), group in df.groupby(
        ["jour_semaine", "jour_num", "langue"], dropna=True
    ):
        rows.append(aggregate_group(group, {
            "jour_semaine": jour,
            "jour_num":     int(jour_num),
            "langue":       langue,
        }))
    result = pd.DataFrame(rows).sort_values(["jour_num", "langue"]).reset_index(drop=True)
    log.info(f"Day of week: {len(result):,} rows")
    return result


def aggregate_global(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating globally by langue + forum...")
    rows = []
    for (langue, forum), group in df.groupby(["langue", "forum"], dropna=True):
        rows.append(aggregate_group(group, {"langue": langue, "forum": forum}))
    result = pd.DataFrame(rows).sort_values(["langue", "forum"]).reset_index(drop=True)
    log.info(f"Global: {len(result):,} rows")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregation temporelle des indicateurs NLP")
    parser.add_argument("--input",      type=Path, default=Path("data/processed/reddit_emotions.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/aggregated"))
    parser.add_argument("--sample",     type=int,  default=None)
    args = parser.parse_args()

    log.info("=== Starting aggregation ===")

    # Charger le fichier le plus enrichi disponible
    topics_path = args.input.parent / "reddit_topics.csv"
    if topics_path.exists():
        log.info(f"Topics file found → {topics_path}")
        df = pd.read_csv(topics_path)
    else:
        log.info(f"No topics file → {args.input}")
        df = pd.read_csv(args.input)

    log.info(f"Loaded {len(df):,} posts")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        log.info(f"Sampling {args.sample} posts")

    # Convertir booléens
    for col in ["is_empty", "is_too_short", "langue_conflit"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Exports
    exports = [
        ("agg_monthly.csv",   aggregate_by_month(df)),
        ("agg_seasonal.csv",  aggregate_by_saison(df)),
        ("agg_dayofweek.csv", aggregate_by_day(df)),
        ("agg_global.csv",    aggregate_global(df)),
    ]

    for filename, result in exports:
        path = args.output_dir / filename
        result.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"Saved → {path} ({len(result)} rows)")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()