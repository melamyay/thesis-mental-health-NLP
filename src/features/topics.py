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

# Modèle multilingue léger FR+EN
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

NR_TOPICS       = 12    # nombre cible de topics (-1 = automatique)
MIN_TOPIC_SIZE  = 20    # nb minimum de posts par topic
NR_TOP_WORDS    = 10    # nb de mots représentatifs par topic

# Thèmes attendus pour guider l'interprétation (pas des seeds durs)
EXPECTED_THEMES = [
    "stress_academique", "sante_mentale", "carriere_emploi",
    "vie_sociale", "finances", "sommeil_fatigue",
    "orientation_choix", "examens_notes", "detresse_crise", "vie_quotidienne"
]


# ── BERTopic pipeline ─────────────────────────────────────────────────────────

def build_bertopic_model(nr_topics: int, min_topic_size: int):
    """
    BERTopic avec :
    - paraphrase-multilingual-MiniLM-L12-v2 : embeddings FR+EN légers
    - UMAP : réduction dimensionnelle (meilleur que PCA pour clusters)
    - HDBSCAN : clustering density-based
    - CountVectorizer : représentation des topics
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words=None,  # stopwords déjà retirés au preprocessing
    )

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=nr_topics,
        top_n_words=NR_TOP_WORDS,
        calculate_probabilities=True,
        verbose=True,
    )

    return model


def run_bertopic(df: pd.DataFrame, nr_topics: int, min_topic_size: int):
    df = df.copy()

    # Filtrer EN/FR uniquement
    before = len(df)
    df = df[df["langue"].isin(["en", "fr"])].copy().reset_index(drop=True)
    log.info(f"Filtered EN/FR: {before} → {len(df)} posts")

    # Filtrer posts trop courts
    df = df[df["nb_tokens"] >= 5].copy().reset_index(drop=True)
    log.info(f"After min_tokens filter: {len(df)} posts")

    # Texte : texte_lemmatise (tokens nettoyés) avec fallback texte_clean
    texts = df["texte_lemmatise"].fillna("").astype(str).tolist()
    texts = [t if t.strip() else str(df["texte_clean"].iloc[i])
             for i, t in enumerate(texts)]

    log.info(f"Running BERTopic on {len(texts):,} documents — target {nr_topics} topics")
    model = build_bertopic_model(nr_topics, min_topic_size)

    topics, probs = model.fit_transform(texts)
    df["topic_id"] = topics

    # Probabilité maximale par post
    import numpy as np
    if hasattr(probs, '__iter__') and len(probs) > 0 and hasattr(probs[0], '__iter__'):
        df["topic_prob"] = [round(float(p.max()), 4) for p in probs]
    else:
        df["topic_prob"] = [round(float(p), 4) for p in probs]

    # Labels lisibles : top 3 mots du topic
    topic_info = model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_labels[tid] = "outlier"
        else:
            words = model.get_topic(tid)
            if words:
                label = " | ".join([w for w, _ in words[:3]])
                topic_labels[tid] = f"T{tid}: {label}"
            else:
                topic_labels[tid] = f"T{tid}"

    df["topic_label"] = df["topic_id"].map(topic_labels)

    # Stats
    log.info("\nTopic distribution:")
    log.info(topic_info[["Topic", "Count", "Name"]].to_string(index=False))
    log.info(f"\nOutliers (topic -1): {(df['topic_id'] == -1).sum():,} posts")
    log.info(f"Posts assigned: {(df['topic_id'] != -1).sum():,} posts")
    log.info(f"Mean topic probability: {df['topic_prob'].mean():.3f}")

    return df, model, topic_info


def save_topic_report(topic_info, model, output_dir: Path):
    """Sauvegarde le rapport détaillé des topics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words = model.get_topic(tid)
        top_words = ", ".join([f"{w} ({round(s, 3)})" for w, s in words[:NR_TOP_WORDS]])
        rows.append({
            "topic_id":  tid,
            "count":     row["Count"],
            "top_words": top_words,
        })
    report_path = output_dir / "topic_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    log.info(f"Topic report → {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Topic modelling — BERTopic multilingue FR+EN"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/reddit_emotions.csv"),
        help="CSV enrichi avec sentiment + émotions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reddit_topics.csv"),
        help="CSV de sortie avec topic_id, topic_prob, topic_label"
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=Path("reports/topics"),
        help="Dossier pour le rapport CSV des topics"
    )
    parser.add_argument(
        "--nr_topics",
        type=int,
        default=NR_TOPICS,
        help="Nombre cible de topics (-1 = automatique)"
    )
    parser.add_argument(
        "--min_topic_size",
        type=int,
        default=MIN_TOPIC_SIZE,
        help="Nombre minimum de posts par topic"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Tester sur N posts seulement"
    )
    args = parser.parse_args()

    log.info("=== Starting BERTopic topic modelling ===")
    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df):,} posts")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        log.info(f"Sampling {args.sample} posts")

    df, model, topic_info = run_bertopic(df, args.nr_topics, args.min_topic_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info(f"Exported {len(df):,} rows → {args.output}")

    save_topic_report(topic_info, model, args.report_dir)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()