"""
main.py

Entry point for the full NLP pipeline.

Usage:
    python main.py
    python main.py --input_dir data/raw/forums_reddit --output_dir data/processed
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def run_preprocessing(input_dir: Path, output_dir: Path):
    from src.preprocessing.preprocess import load_and_merge, clean, export
    log.info("── Step 1: Preprocessing ──")
    df = load_and_merge(input_dir)
    df = clean(df)
    output_path = output_dir / "reddit_clean.csv"
    export(df, output_path)
    return output_path


def run_nlp(input_path: Path, output_dir: Path):
    # TODO: implement NLP analysis (sentiment, topic modeling, etc.)
    log.info("── Step 2: NLP Analysis ── (not yet implemented)")
    pass


def run_knowledge_graph(input_path: Path):
    # TODO: implement knowledge graph construction (Neo4j)
    log.info("── Step 3: Knowledge Graph ── (not yet implemented)")
    pass


def main():
    parser = argparse.ArgumentParser(description="Run the full mental health NLP pipeline")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw/forums_reddit"),
        help="Directory containing raw XLSX files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed outputs"
    )
    args = parser.parse_args()

    log.info("════════════════════════════════════════")
    log.info("   thesis-mental-health-NLP pipeline    ")
    log.info("════════════════════════════════════════")

    # Step 1 — Preprocessing
    cleaned_path = run_preprocessing(args.input_dir, args.output_dir)

    # Step 2 — NLP Analysis
    run_nlp(cleaned_path, args.output_dir)

    # Step 3 — Knowledge Graph
    run_knowledge_graph(cleaned_path)

    log.info("════════════════════════════════════════")
    log.info("   Pipeline complete ✓                  ")
    log.info("════════════════════════════════════════")


if __name__ == "__main__":
    main()