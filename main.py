"""
main.py

Entry point for the full NLP pipeline.

Usage:
    python main.py      # run full pipeline
    python main.py --skip_nlp       # preprocessing only
    python main.py --input_dir data/raw     # custom input dir
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Step 1 : Preprocessing ────────────────────────────────────────────────────

def run_preprocessing(input_dir: Path, output_dir: Path) -> Path:
    from src.preprocessing.preprocess import load_and_merge, clean, export, load_spacy_models
    log.info("── Step 1: Preprocessing ──")
    nlp_models = load_spacy_models()
    df = load_and_merge(input_dir)
    df = clean(df, nlp_models)
    output_path = output_dir / "reddit_clean.csv"
    export(df, output_path)
    return output_path

# ── Step 2 : Sentiment ────────────────────────────────────────────────────────

def run_sentiment(input_path: Path, output_dir: Path) -> Path:
    from src.features.sentiment import extract_sentiment
    import pandas as pd
    log.info("── Step 2: Sentiment extraction ──")
    df = pd.read_csv(input_path)
    df = extract_sentiment(df)
    output_path = output_dir / "reddit_sentiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info(f"Sentiment saved → {output_path}")
    return output_path
 
# ── Step 3 : Emotions + Distress ─────────────────────────────────────────────
 
def run_emotions(input_path: Path, output_dir: Path) -> Path:
    from src.features.emotions import extract_emotions
    import pandas as pd
    log.info("── Step 3: Emotion extraction + distress index ──")
    df = pd.read_csv(input_path)
    df = extract_emotions(df)
    output_path = output_dir / "reddit_emotions.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info(f"Emotions saved → {output_path}")
    return output_path
 
 # ── Step 4 : Topic modelling ──────────────────────────────────────────────────
 
def run_topics(input_path: Path, output_dir: Path) -> Path:
    from src.features.topics import run_bertopic, save_topic_report
    import pandas as pd
    log.info("── Step 4: Topic modelling (BERTopic) ──")
    df = pd.read_csv(input_path)
    df, model, topic_info = run_bertopic(df, nr_topics=12, min_topic_size=20)
    output_path = output_dir / "reddit_topics.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_topic_report(topic_info, model, Path("reports/topics"))
    log.info(f"Topics saved → {output_path}")
    return output_path
 
 # ── Step 5 : Temporal aggregation ────────────────────────────────────────────
 
def run_aggregation(input_path: Path, output_dir: Path):
    from src.features.aggregation import (
        aggregate_by_month, aggregate_by_saison,
        aggregate_by_day, aggregate_global
    )
    import pandas as pd
    log.info("── Step 5: Temporal aggregation ──")
    df = pd.read_csv(input_path)
 
    # Convertir booléens
    for col in ["is_empty", "is_too_short", "langue_conflit"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(False)
 
    agg_dir = output_dir.parent / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
 
    aggregate_by_month(df).to_csv(agg_dir / "agg_monthly.csv",   index=False, encoding="utf-8-sig")
    aggregate_by_saison(df).to_csv(agg_dir / "agg_seasonal.csv", index=False, encoding="utf-8-sig")
    aggregate_by_day(df).to_csv(agg_dir / "agg_dayofweek.csv",   index=False, encoding="utf-8-sig")
    aggregate_global(df).to_csv(agg_dir / "agg_global.csv",      index=False, encoding="utf-8-sig")
    log.info(f"Aggregations saved → {agg_dir}")

# ── Step 6 : Visualizations ───────────────────────────────────────────────────
 
def run_visualizations(input_path: Path):
    import sys
    sys.argv = ["visualize.py"]  # reset argv pour éviter les conflits argparse
    from src.features.visualize import (
        fig_corpus_overview, fig_sentiment_global, fig_emotions_global,
        fig_distress, fig_emotions_temporal, fig_emotions_heatmap,
        fig_distress_saison, fig_sentiment_dayofweek, fig_high_distress_posts,
        FIGURES_DIR
    )
    import pandas as pd
    log.info("── Step 6: Visualizations ──")
    df = pd.read_csv(input_path)
    for col in ["is_empty", "is_too_short"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_corpus_overview(df)
    fig_sentiment_global(df)
    fig_emotions_global(df)
    fig_distress(df)
    fig_emotions_temporal(df)
    fig_emotions_heatmap(df)
    fig_distress_saison(df)
    fig_sentiment_dayofweek(df)
    fig_high_distress_posts(df)
    log.info(f"Figures saved → {FIGURES_DIR}")

# ── Step 7 : Graphs ───────────────────────────────────────────────────────────
 
def run_graphs(input_path: Path):
    from src.visualization.graph import (
        build_cooc_graph, build_topic_graph,
        export_pyvis, export_png, setup_dirs
    )
    import pandas as pd
    log.info("── Step 7: Graph generation ──")
    df = pd.read_csv(input_path)
    setup_dirs()
    for lang in ["en", "fr"]:
        G, wf = build_cooc_graph(df, lang=lang)
        if G.number_of_nodes() > 0:
            export_pyvis(G, f"cooc_{lang}.html", word_freq=wf)
            export_png(G, f"cooc_{lang}.png",
                       title=f"Co-occurrences ({lang.upper()})", word_freq=wf)
    G_all, wf_all = build_cooc_graph(df, lang="all")
    export_pyvis(G_all, "cooc_all.html", word_freq=wf_all)
    export_png(G_all, "cooc_all.png", title="Co-occurrences (EN+FR)", word_freq=wf_all)
    G_topics = build_topic_graph(df)
    if G_topics.number_of_nodes() > 0:
        export_pyvis(G_topics, "topics_words.html")
        export_png(G_topics, "topics_words.png", title="Topics ↔ Mots clés")
        log.info("Graphs saved → reports/graphs/")


        # ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the full mental health NLP pipeline")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Dossier contenant les dossiers forums_reddit* (scannés automatiquement)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Dossier de sortie pour les fichiers traités"
    )
    parser.add_argument(
        "--skip_topics",
        action="store_true",
        help="Ignorer BERTopic (utile sur Mac sans llvmlite)"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("════════════════════════════════════════")
    log.info("   thesis-mental-health-NLP pipeline    ")
    log.info("════════════════════════════════════════")

    # Step 1 — Preprocessing
    cleaned_path = run_preprocessing(args.input_dir, args.output_dir)
 
    if args.skip_nlp:
        log.info("--skip_nlp flag set, stopping after preprocessing.")
    else:
        # Step 2 — Sentiment
        sentiment_path = run_sentiment(cleaned_path, args.output_dir)
 
        # Step 3 — Emotions + distress
        emotions_path = run_emotions(sentiment_path, args.output_dir)
 
        # Step 4 — Topics
        if args.skip_topics:
            log.info("-- skip_topics flag set, skipping BERTopic.")
            topics_path = emotions_path
        else:
            topics_path = run_topics(emotions_path, args.output_dir)

        # Step 5 — Aggregation
        run_aggregation(topics_path, args.output_dir)
 
        # Step 6 — Visualizations
        run_visualizations(topics_path)
 
        # Step 7 — Graphs
        run_graphs(topics_path)
 
    log.info("════════════════════════════════════════")
    log.info("   Pipeline complete ✓                  ")
    log.info("════════════════════════════════════════")


if __name__ == "__main__":
    main()