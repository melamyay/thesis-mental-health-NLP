import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "fr": "cmarkea/distilcamembert-base-sentiment",
    "en": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}

# Map each model's labels to a unified (negative, neutral, positive) scheme
LABEL_MAP = {
    # distilcamembert outputs 1 star → 5 stars
    "cmarkea/distilcamembert-base-sentiment": {
        "1 star":  "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive",
    },
    # twitter-roberta outputs negative / neutral / positive directly
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        "negative": "negative",
        "neutral":  "neutral",
        "positive": "positive",
    },
}

BATCH_SIZE = 8      # safe for CPU
MAX_LENGTH = 512     # transformer max tokens
DEVICE     = -1      # -1 = CPU


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_pipeline(model_name: str):
    """Load a HuggingFace sentiment pipeline."""
   # Import ici pour éviter les conflits NumPy au module level
    import torch  # noqa: F401
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    log.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        truncation=True,
        max_length=MAX_LENGTH,
        top_k=None,   # return all label scores
    )


def _scores_to_dict(raw_scores: list, model_name: str) -> dict:

    label_map = LABEL_MAP[model_name]

    # Aggregate scores by unified label
    unified = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for item in raw_scores:
        unified_label = label_map.get(item["label"].lower(), "neutral")
        unified[unified_label] += item["score"]

    # Predicted label = highest score
    predicted = max(unified, key=unified.get)

    # Continuous score: positive anchored at +1, negative at -1
    score_cont = unified["positive"] - unified["negative"]

    return {
        "sentiment_label": predicted,
        "score_pos":       round(unified["positive"], 4),
        "score_neu":       round(unified["neutral"],  4),
        "score_neg":       round(unified["negative"], 4),
        "score_cont":      round(score_cont, 4),
    }


def _run_batch(texts: list[str], pipe, model_name: str) -> list[dict]:
    """Run inference on a batch of texts and return unified score dicts."""
    # Replace empty strings with a neutral placeholder to avoid pipeline errors
    safe_texts = [t if t and t.strip() else "." for t in texts]
    try:
        raw_outputs = pipe(safe_texts, batch_size=BATCH_SIZE)
    except Exception as e:
        log.warning(f"Batch inference error: {e} — filling with NaN")
        return [{"sentiment_label": None, "score_pos": None,
                 "score_neu": None, "score_neg": None, "score_cont": None}] * len(texts)

    return [_scores_to_dict(output, model_name) for output in raw_outputs]


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment columns to the dataframe.
    Processes EN and FR posts separately with their respective models.
    Posts with langue == 'other' or 'unknown' are processed with the EN model.
    """
    import torch
    df = df.copy()

    # Init result columns
    for col in ["sentiment_label", "score_pos", "score_neu", "score_neg", "score_cont"]:
        df[col] = None

    for lang in ["fr", "en"]:
        model_name = MODELS[lang]

        # FR → fr model | EN + other + unknown → en model
        if lang == "fr":
            mask = df["langue"] == "fr"
        else:
            mask = df["langue"] != "fr"

        subset = df[mask].copy()
        if subset.empty:
            log.info(f"No posts for lang={lang}, skipping.")
            continue

        log.info(f"Running sentiment for lang={lang} — {len(subset)} posts")
        pipe = _build_pipeline(model_name)

        texts = subset["texte_clean"].fillna("").tolist()
        results = []

        # Process in batches with progress logging
        total = len(texts)
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            results.extend(_run_batch(batch, pipe, model_name))
            done = min(i + BATCH_SIZE, total)
            if done % 80 == 0 or done == total:
                log.info(f"  [{lang}] {done}/{total} posts processed")

        # Write results back
        result_df = pd.DataFrame(results, index=subset.index)
        for col in result_df.columns:
            df.loc[mask, col] = result_df[col]

        # Free memory
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Stats
    log.info("Sentiment distribution:")
    log.info(df["sentiment_label"].value_counts().to_string())
    log.info(f"Average continuous score: {df['score_cont'].mean():.3f}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sentiment extraction for Reddit posts")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/reddit_clean.csv"),
        help="Path to cleaned CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reddit_sentiment.csv"),
        help="Output CSV path with sentiment scores"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Nombre de posts à traiter (ex: --sample 100 pour tester)"
    )
    args = parser.parse_args()

    log.info("=== Starting sentiment extraction ===")
    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df):,} posts")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    df = extract_sentiment(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info(f"Exported {len(df):,} rows → {args.output}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()