import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MOIS_LABELS = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Aoû",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
}

PALETTE = {
    "en":      "#378ADD",
    "fr":      "#1D9E75",
    "other":   "#EF9F27",
    "unknown": "#888780",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, name: str) -> Path:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved figure: {out}")
    return out


def _pct(n: int, total: int) -> str:
    return f"{n:,} ({n / total * 100:.1f}%)"


# ── Report sections ───────────────────────────────────────────────────────────

def report_overview(df: pd.DataFrame) -> dict:
    total = len(df)
    stats = {
        "total_posts":        total,
        "posts_vides":        _pct(df["is_empty"].sum(), total),
        "posts_trop_courts":  _pct(df["is_too_short"].sum(), total),
        "conflits_langue":    _pct(df["langue_conflit"].sum(), total) if "langue_conflit" in df.columns else "N/A",
        "moy_tokens":         f"{df['nb_tokens'].mean():.1f}",
        "median_tokens":      f"{df['nb_tokens'].median():.0f}",
        "moy_longueur_clean": f"{df['longueur_texte_clean'].mean():.0f} chars",
        "annees_couvertes":   f"{int(df['annee_post'].min())} → {int(df['annee_post'].max())}"
        if df["annee_post"].notna().any() else "N/A",
    }
    return stats


def plot_langue_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["langue"].value_counts()
    colors = [PALETTE.get(lang, "#B4B2A9") for lang in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=10
        )

    ax.set_title("Distribution des langues", fontsize=13, pad=12)
    ax.set_xlabel("Langue")
    ax.set_ylabel("Nombre de posts")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _savefig(fig, out_dir, "langue_distribution.png")


def plot_forum_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["forum"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#5DCAA5", edgecolor="white")

    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(
            val + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9
        )

    ax.set_title("Distribution par forum (subreddit)", fontsize=13, pad=12)
    ax.set_xlabel("Nombre de posts")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _savefig(fig, out_dir, "forum_distribution.png")


def plot_timeline(df: pd.DataFrame, out_dir: Path) -> Path:
    """Posts per month over time, colored by language."""
    if "annee_post" not in df.columns or df["annee_post"].isna().all():
        log.warning("No date data available, skipping timeline plot.")
        return None

    df = df.copy()
    df["periode"] = pd.to_datetime(
        df["annee_post"].astype(str) + "-" + df["mois_post"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
    df = df.dropna(subset=["periode"])

    pivot = (
        df.groupby(["periode", "langue"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    for lang in pivot.columns:
        color = PALETTE.get(lang, "#B4B2A9")
        ax.plot(pivot.index, pivot[lang], label=lang, color=color, linewidth=1.8, marker="o", markersize=3)

    ax.set_title("Volume de posts par mois et par langue", fontsize=13, pad=12)
    ax.set_xlabel("Période")
    ax.set_ylabel("Nombre de posts")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Langue", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _savefig(fig, out_dir, "timeline_posts.png")


def plot_token_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram all posts
    axes[0].hist(df["nb_tokens"], bins=50, color="#7F77DD", edgecolor="white", linewidth=0.5)
    axes[0].axvline(5, color="#E24B4A", linestyle="--", linewidth=1.2, label="Seuil min (5 tokens)")
    axes[0].set_title("Distribution du nb de tokens", fontsize=12)
    axes[0].set_xlabel("Nombre de tokens")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Boxplot by language
    langs = sorted(df["langue"].unique())
    data_by_lang = [df[df["langue"] == l]["nb_tokens"].values for l in langs]
    bp = axes[1].boxplot(data_by_lang, labels=langs, patch_artist=True, medianprops={"color": "white", "linewidth": 2})
    for patch, lang in zip(bp["boxes"], langs):
        patch.set_facecolor(PALETTE.get(lang, "#B4B2A9"))
    axes[1].set_title("Tokens par langue", fontsize=12)
    axes[1].set_xlabel("Langue")
    axes[1].set_ylabel("Nombre de tokens")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _savefig(fig, out_dir, "token_distribution.png")


def plot_quality_flags(df: pd.DataFrame, out_dir: Path) -> Path:
    total = len(df)
    flags = {
        "Posts valides":      total - df["is_empty"].sum() - df["is_too_short"].sum(),
        "Trop courts\n(<5 tokens)": df["is_too_short"].sum(),
        "Vides":              df["is_empty"].sum(),
    }
    if "langue_conflit" in df.columns:
        flags["Conflit\nde langue"] = df["langue_conflit"].sum()

    labels = list(flags.keys())
    values = list(flags.values())
    colors = ["#1D9E75", "#EF9F27", "#E24B4A", "#7F77DD"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Qualité du dataset", fontsize=13, pad=12)
    fig.tight_layout()
    return _savefig(fig, out_dir, "quality_flags.png")


def print_report(stats: dict, df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  RAPPORT DE VALIDATION — DATASET NETTOYÉ")
    print("=" * 55)

    print("\n── Vue d'ensemble ──────────────────────────────────")
    for k, v in stats.items():
        label = k.replace("_", " ").capitalize()
        print(f"  {label:<28} {v}")

    print("\n── Distribution par forum ──────────────────────────")
    forum_counts = df["forum"].value_counts()
    for forum, count in forum_counts.items():
        print(f"  {forum:<20} {_pct(count, len(df))}")

    print("\n── Distribution des langues ────────────────────────")
    lang_counts = df["langue"].value_counts()
    for lang, count in lang_counts.items():
        print(f"  {lang:<20} {_pct(count, len(df))}")

    print("\n── Source du texte ─────────────────────────────────")
    source_counts = df["texte_source"].value_counts()
    for src, count in source_counts.items():
        print(f"  {src:<20} {_pct(count, len(df))}")

    if "annee_post" in df.columns:
        print("\n── Volume par année ────────────────────────────────")
        year_counts = df["annee_post"].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {int(year):<20} {_pct(count, len(df))}")

    print("\n" + "=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validation report for cleaned Reddit dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/reddit_clean.csv"),
        help="Path to cleaned CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to save figures"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip figure generation (text report only)"
    )
    args = parser.parse_args()

    log.info(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df):,} rows")

    # Parse booleans if stored as strings
    for col in ["is_empty", "is_too_short", "langue_conflit"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    stats = report_overview(df)
    print_report(stats, df)

    if not args.no_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving figures to: {args.output_dir}")
        plot_langue_distribution(df, args.output_dir)
        plot_forum_distribution(df, args.output_dir)
        plot_timeline(df, args.output_dir)
        plot_token_distribution(df, args.output_dir)
        plot_quality_flags(df, args.output_dir)
        log.info("All figures saved.")

    log.info("=== Validation done ===")


if __name__ == "__main__":
    main()