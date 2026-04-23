import argparse
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GOEMO_MODEL = "SamLowe/roberta-base-go_emotions"

NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust"]

NRC_CSV_PATH = Path("data/lexicons/nrc_lexicon.csv")
NRC_URL = "https://raw.githubusercontent.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

GOEMO_TO_NRC = {
    "anger":          "anger",
    "annoyance":      "anger",
    "disapproval":    "anger",
    "disgust":        "disgust",
    "fear":           "fear",
    "nervousness":    "fear",
    "joy":            "joy",
    "amusement":      "joy",
    "excitement":     "joy",
    "optimism":       "anticipation",
    "anticipation":   "anticipation",
    "surprise":       "surprise",
    "realization":    "surprise",
    "sadness":        "sadness",
    "grief":          "sadness",
    "disappointment": "sadness",
    "remorse":        "sadness",
    "trust":          "trust",
    "admiration":     "trust",
    "approval":       "trust",
    "caring":         "trust",
    "curiosity":      "anticipation",
    "desire":         "anticipation",
    "love":           "joy",
    "pride":          "joy",
    "relief":         "joy",
    "confusion":      "surprise",
    "neutral":        None,
}

NRC_WEIGHT  = 0.30
BERT_WEIGHT = 0.70
BATCH_SIZE  = 8
MAX_LENGTH  = 512
DEVICE      = -1


# ── NRC CSV lexicon ───────────────────────────────────────────────────────────

def load_nrc_csv() -> dict:
    """
    Load NRC lexicon as word → set of emotions.
    Downloads once to data/lexicons/nrc_lexicon.csv if not present.
    """
    NRC_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not NRC_CSV_PATH.exists():
        log.info(f"Downloading NRC lexicon → {NRC_CSV_PATH}")
        try:
            import ssl
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(NRC_URL, context=ctx) as r:
                content = r.read().decode("utf-8")
            with open(NRC_CSV_PATH, "w") as f:
                f.write(content)
            log.info("NRC lexicon downloaded")
        except Exception as e:
            log.error(f"Download failed: {e}")
            log.error("Download manually: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm")
            log.error(f"Place file at: {NRC_CSV_PATH}")
            return {}

    try:
        df = pd.read_csv(NRC_CSV_PATH, sep="\t", header=None,
                         names=["word", "emotion", "association"])
        df = df[(df["association"] == 1) & (df["emotion"].isin(NRC_EMOTIONS))]
        lexicon = {}
        for _, row in df.iterrows():
            word = row["word"].lower()
            if word not in lexicon:
                lexicon[word] = set()
            lexicon[word].add(row["emotion"])
        log.info(f"NRC lexicon loaded: {len(lexicon):,} words")
        return lexicon
    except Exception as e:
        log.error(f"Failed to parse NRC lexicon: {e}")
        return {}


def score_nrc(tokens_str: str, lexicon: dict) -> dict:
    """Score a post via NRC lexicon. Input = texte_lemmatise (space-separated tokens)."""
    empty = {f"nrc_{e}": 0.0 for e in NRC_EMOTIONS}
    empty["nrc_dominant"] = None

    if not lexicon or not tokens_str or not str(tokens_str).strip():
        return empty

    tokens = str(tokens_str).lower().split()
    if not tokens:
        return empty

    counts = {e: 0 for e in NRC_EMOTIONS}
    for token in tokens:
        if token in lexicon:
            for emotion in lexicon[token]:
                counts[emotion] += 1

    if sum(counts.values()) == 0:
        return empty

    scores = {f"nrc_{e}": round(counts[e] / len(tokens), 4) for e in NRC_EMOTIONS}
    best = max(NRC_EMOTIONS, key=lambda e: scores[f"nrc_{e}"])
    scores["nrc_dominant"] = best if scores[f"nrc_{best}"] > 0 else None
    return scores


# ── go-emotions BERT ──────────────────────────────────────────────────────────

def load_goemo_pipeline():
    from transformers import pipeline
    log.info(f"Loading model: {GOEMO_MODEL}")
    return pipeline(
        "text-classification",
        model=GOEMO_MODEL,
        device=DEVICE,
        truncation=True,
        max_length=MAX_LENGTH,
        top_k=None,
    )


def score_goemo_batch(texts: list, pipe) -> list:
    safe = [t if t and t.strip() else "." for t in texts]
    empty = {f"goemo_{e}": 0.0 for e in NRC_EMOTIONS}
    empty["goemo_dominant"] = None
    try:
        outputs = pipe(safe, batch_size=BATCH_SIZE)
    except Exception as ex:
        log.warning(f"go-emotions batch error: {ex}")
        return [empty.copy() for _ in texts]

    results = []
    for output in outputs:
        mapped = {e: 0.0 for e in NRC_EMOTIONS}
        for item in output:
            nrc_label = GOEMO_TO_NRC.get(item["label"])
            if nrc_label:
                mapped[nrc_label] += item["score"]
        total = sum(mapped.values())
        if total > 0:
            mapped = {e: round(v / total, 4) for e, v in mapped.items()}
        dominant = max(NRC_EMOTIONS, key=lambda e: mapped[e])
        row = {f"goemo_{e}": mapped[e] for e in NRC_EMOTIONS}
        row["goemo_dominant"] = dominant if mapped[dominant] > 0 else None
        results.append(row)
    return results


# ── Combined scores + distress index ─────────────────────────────────────────

def compute_combined_scores(df: pd.DataFrame, has_bert: bool) -> pd.DataFrame:
    """
    combined_{e} = 0.30 * nrc_{e} + 0.70 * goemo_{e}  (si BERT dispo)
                 = nrc_{e}                              (sinon)

    distress_score = (fear + sadness + 0.5*anger) / 2.5  → [0,1]
    fear est le proxy anxiety (fear + nervousness dans go-emotions)
    """
    log.info("Computing combined scores and distress index...")

    for e in NRC_EMOTIONS:
        if has_bert:
            df[f"emotion_{e}"] = (
                NRC_WEIGHT  * df[f"nrc_{e}"].fillna(0) +
                BERT_WEIGHT * df[f"goemo_{e}"].fillna(0)
            ).round(4)
        else:
            df[f"emotion_{e}"] = df[f"nrc_{e}"].fillna(0).round(4)

    emotion_cols = [f"emotion_{e}" for e in NRC_EMOTIONS]
    df["emotion_dominant"] = df[emotion_cols].idxmax(axis=1).str.replace("emotion_", "")
    df.loc[df[emotion_cols].max(axis=1) == 0, "emotion_dominant"] = None

    df["distress_score"] = (
        1.0 * df["emotion_fear"].fillna(0) +
        1.0 * df["emotion_sadness"].fillna(0) +
        0.5 * df["emotion_anger"].fillna(0)
    ).div(2.5).round(4)

    df["distress_level"] = pd.cut(
        df["distress_score"],
        bins=[0, 0.2, 0.4, 0.6, 1.01],
        labels=["low", "moderate", "high", "severe"],
        include_lowest=True
    )

    log.info(f"Mean distress score: {df['distress_score'].mean():.3f}")
    log.info("Distress level distribution:")
    log.info(df["distress_level"].value_counts().sort_index().to_string())
    log.info("Dominant emotion distribution:")
    log.info(df["emotion_dominant"].value_counts().to_string())
    return df


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_emotions(df: pd.DataFrame, skip_bert: bool = False) -> pd.DataFrame:
    df = df.copy()

    for e in NRC_EMOTIONS:
        df[f"nrc_{e}"] = 0.0
    df["nrc_dominant"] = None

    if not skip_bert:
        for e in NRC_EMOTIONS:
            df[f"goemo_{e}"] = 0.0
        df["goemo_dominant"] = None

    # 1. NRC
    log.info("Scoring with NRC lexicon (CSV)...")
    lexicon = load_nrc_csv()
    if lexicon:
        nrc_results = df["texte_lemmatise"].apply(lambda t: score_nrc(t, lexicon))
        nrc_df = pd.DataFrame(nrc_results.tolist(), index=df.index)
        for col in nrc_df.columns:
            df[col] = nrc_df[col]
        log.info("NRC dominant distribution:")
        log.info(df["nrc_dominant"].value_counts().to_string())

    # 2. go-emotions BERT
    has_bert = False
    if not skip_bert:
        log.info("Scoring with go-emotions BERT...")
        pipe = load_goemo_pipeline()
        texts = df["texte_clean"].fillna("").tolist()
        total = len(texts)
        results = []
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            results.extend(score_goemo_batch(batch, pipe))
            done = min(i + BATCH_SIZE, total)
            if done % 160 == 0 or done == total:
                log.info(f"  [go-emotions] {done}/{total} posts processed")
        goemo_df = pd.DataFrame(results, index=df.index)
        for col in goemo_df.columns:
            df[col] = goemo_df[col]
        del pipe
        has_bert = True

    # 3. Combined + distress
    df = compute_combined_scores(df, has_bert=has_bert)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emotion extraction — NRC CSV + go-emotions BERT + distress")
    parser.add_argument("--input",   type=Path, default=Path("data/processed/reddit_sentiment.csv"))
    parser.add_argument("--output",  type=Path, default=Path("data/processed/reddit_emotions.csv"))
    parser.add_argument("--sample",  type=int,  default=None)
    parser.add_argument("--no_bert", action="store_true", help="NRC only, mode rapide")
    args = parser.parse_args()

    log.info("=== Starting emotion extraction ===")
    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df):,} posts")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        log.info(f"Sampling {args.sample} posts")

    df = extract_emotions(df, skip_bert=args.no_bert)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info(f"Exported {len(df):,} rows → {args.output}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
