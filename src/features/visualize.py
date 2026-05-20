"""
visualize.py — Génère toutes les figures d'analyse NLP
Sortie : reports/figures/
Usage  : python3 src/features/visualize.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

FIGURES_DIR = Path("reports/figures")

SAISON_MAP = {
    1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps",
    5: "examens", 6: "ete", 7: "ete", 8: "ete",
    9: "rentree", 10: "automne", 11: "automne", 12: "examens",
}
SAISON_ORDER = ["rentree", "automne", "hiver", "printemps", "examens", "ete"]
MOIS_LABELS  = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
                7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}

EMOTION_COLS = ["emotion_anger","emotion_anticipation","emotion_disgust",
                "emotion_fear","emotion_joy","emotion_sadness",
                "emotion_surprise","emotion_trust"]

EMOTION_COLORS = {
    "anger":        "#E24B4A",
    "anticipation": "#BA7517",
    "disgust":      "#712B13",
    "fear":         "#A32D2D",
    "joy":          "#1D9E75",
    "sadness":      "#534AB7",
    "surprise":     "#EF9F27",
    "trust":        "#085041",
}

SENTIMENT_COLORS = {"positive": "#1D9E75", "neutral": "#888780", "negative": "#E24B4A"}
FORUM_COLORS     = {"en": "#378ADD", "fr": "#1D9E75", "other": "#BA7517"}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 1. Vue d'ensemble corpus ──────────────────────────────────────────────────

def fig_corpus_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Vue d'ensemble du corpus Reddit", fontsize=13, y=1.02)

    # Distribution par forum
    counts = df["forum"].value_counts()
    axes[0].barh(counts.index, counts.values, color="#378ADD", edgecolor="white")
    axes[0].set_title("Posts par subreddit")
    axes[0].set_xlabel("Nombre de posts")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 5, i, f"{v:,}", va="center", fontsize=8)

    # Distribution par langue
    lang_counts = df["langue"].value_counts()
    colors = [FORUM_COLORS.get(l, "#888780") for l in lang_counts.index]
    axes[1].bar(lang_counts.index, lang_counts.values, color=colors, edgecolor="white", width=0.5)
    axes[1].set_title("Distribution des langues")
    axes[1].set_ylabel("Nombre de posts")
    for i, (lang, v) in enumerate(lang_counts.items()):
        axes[1].text(i, v + 10, f"{v:,}", ha="center", fontsize=9)

    # Distribution des posts par mois
    monthly = df.groupby("mois_post").size()
    axes[2].bar([MOIS_LABELS.get(m, m) for m in monthly.index], monthly.values,
                color="#5DCAA5", edgecolor="white")
    axes[2].set_title("Volume de posts par mois")
    axes[2].set_ylabel("Nombre de posts")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save(fig, "01_corpus_overview.png")


# ── 2. Sentiment global ───────────────────────────────────────────────────────

def fig_sentiment_global(df):
    if "sentiment_label" not in df.columns:
        log.warning("sentiment_label manquant, skip fig_sentiment_global")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Analyse de sentiment", fontsize=13, y=1.02)

    # Distribution globale
    counts = df["sentiment_label"].value_counts()
    colors = [SENTIMENT_COLORS.get(l, "#888780") for l in counts.index]
    wedges, texts, autotexts = axes[0].pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in autotexts: t.set_fontsize(9)
    axes[0].set_title("Distribution globale")

    # Score continu par langue
    if "score_cont" in df.columns:
        for lang, color in [("en", "#378ADD"), ("fr", "#1D9E75")]:
            subset = df[df["langue"] == lang]["score_cont"].dropna()
            if len(subset) > 0:
                axes[1].hist(subset, bins=40, alpha=0.6, color=color,
                             label=f"{lang} (n={len(subset):,})", edgecolor="none")
        axes[1].axvline(0, color="#E24B4A", linestyle="--", linewidth=1, label="neutre")
        axes[1].set_title("Score continu par langue")
        axes[1].set_xlabel("Score continu [-1, 1]")
        axes[1].legend(fontsize=8)

    # Évolution mensuelle du score continu
    if "score_cont" in df.columns and "mois_post" in df.columns:
        monthly_sent = df.groupby(["mois_post", "langue"])["score_cont"].mean().unstack(fill_value=np.nan)
        for lang in monthly_sent.columns:
            color = FORUM_COLORS.get(lang, "#888780")
            axes[2].plot(
                [MOIS_LABELS.get(m, m) for m in monthly_sent.index],
                monthly_sent[lang],
                marker="o", markersize=4, linewidth=1.8,
                color=color, label=lang
            )
        axes[2].axhline(0, color="#888780", linestyle="--", linewidth=0.8)
        axes[2].set_title("Score sentiment moyen par mois")
        axes[2].set_xlabel("Mois")
        axes[2].set_ylabel("Score continu moyen")
        axes[2].legend(fontsize=8)
        axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save(fig, "02_sentiment_global.png")


# ── 3. Émotions ───────────────────────────────────────────────────────────────

def fig_emotions_global(df):
    available = [c for c in EMOTION_COLS if c in df.columns]
    if not available:
        log.warning("Colonnes émotions manquantes, skip fig_emotions_global")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribution des émotions", fontsize=13, y=1.02)

    # Scores moyens par émotion
    means = {c.replace("emotion_", ""): df[c].mean() for c in available}
    emotions = list(means.keys())
    values   = list(means.values())
    colors   = [EMOTION_COLORS.get(e, "#888780") for e in emotions]

    bars = axes[0].bar(emotions, values, color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("Score moyen par émotion")
    axes[0].set_ylabel("Score moyen")
    axes[0].tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.3f}", ha="center", fontsize=8)

    # Émotion dominante — distribution
    if "emotion_dominant" in df.columns:
        dom_counts = df["emotion_dominant"].value_counts()
        colors_dom = [EMOTION_COLORS.get(e, "#888780") for e in dom_counts.index]
        axes[1].barh(dom_counts.index, dom_counts.values, color=colors_dom, edgecolor="white")
        axes[1].set_title("Émotion dominante par post")
        axes[1].set_xlabel("Nombre de posts")
        for i, v in enumerate(dom_counts.values):
            axes[1].text(v + 2, i, f"{v:,}", va="center", fontsize=8)

    fig.tight_layout()
    save(fig, "03_emotions_global.png")


# ── 4. Distress score ─────────────────────────────────────────────────────────

def fig_distress(df):
    if "distress_score" not in df.columns:
        log.warning("distress_score manquant, skip fig_distress")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Indice de détresse", fontsize=13, y=1.02)

    # Distribution globale
    axes[0].hist(df["distress_score"].dropna(), bins=50,
                 color="#E24B4A", edgecolor="white", linewidth=0.3)
    axes[0].axvline(df["distress_score"].mean(), color="#712B13",
                    linestyle="--", linewidth=1.5, label=f"Moy. {df['distress_score'].mean():.3f}")
    axes[0].set_title("Distribution du distress score")
    axes[0].set_xlabel("Distress score [0, 1]")
    axes[0].legend(fontsize=8)

    # Niveaux de détresse
    if "distress_level" in df.columns:
        level_counts = df["distress_level"].value_counts().reindex(
            ["low", "moderate", "high", "severe"], fill_value=0
        )
        level_colors = ["#1D9E75", "#BA7517", "#E24B4A", "#712B13"]
        bars = axes[1].bar(level_counts.index, level_counts.values,
                           color=level_colors, edgecolor="white", width=0.5)
        axes[1].set_title("Niveaux de détresse")
        axes[1].set_ylabel("Nombre de posts")
        for bar, val in zip(bars, level_counts.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         f"{val:,}", ha="center", fontsize=9)

    # Distress moyen par mois
    if "mois_post" in df.columns:
        monthly = df.groupby(["mois_post", "langue"])["distress_score"].mean().unstack(fill_value=np.nan)
        for lang in monthly.columns:
            color = FORUM_COLORS.get(lang, "#888780")
            axes[2].plot(
                [MOIS_LABELS.get(m, m) for m in monthly.index],
                monthly[lang],
                marker="o", markersize=4, linewidth=1.8,
                color=color, label=lang
            )
        axes[2].set_title("Distress moyen par mois")
        axes[2].set_xlabel("Mois")
        axes[2].set_ylabel("Score moyen")
        axes[2].legend(fontsize=8)
        axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save(fig, "04_distress.png")


# ── 5. Émotions × mois ───────────────────────────────────────────────────────

def fig_emotions_temporal(df):
    available = [c for c in EMOTION_COLS if c in df.columns]
    if not available or "mois_post" not in df.columns:
        return

    monthly = df.groupby("mois_post")[available].mean()
    emotions = [c.replace("emotion_", "") for c in available]

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, emotion in zip(available, emotions):
        ax.plot(
            [MOIS_LABELS.get(m, m) for m in monthly.index],
            monthly[col],
            marker="o", markersize=3, linewidth=1.5,
            color=EMOTION_COLORS.get(emotion, "#888780"),
            label=emotion
        )
    ax.set_title("Évolution mensuelle des émotions", fontsize=12)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Score moyen")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save(fig, "05_emotions_temporal.png")


# ── 6. Heatmap émotions × subreddit ──────────────────────────────────────────

def fig_emotions_heatmap(df):
    available = [c for c in EMOTION_COLS if c in df.columns]
    if not available or "forum" not in df.columns:
        return

    heatmap_data = df.groupby("forum")[available].mean()
    heatmap_data.columns = [c.replace("emotion_", "") for c in heatmap_data.columns]

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.08 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Score moyen par émotion et subreddit", fontsize=12)
    fig.tight_layout()
    save(fig, "06_emotions_heatmap.png")


# ── 7. Distress × saison ─────────────────────────────────────────────────────

def fig_distress_saison(df):
    if "distress_score" not in df.columns or "mois_post" not in df.columns:
        return

    df = df.copy()
    df["saison"] = df["mois_post"].map(SAISON_MAP)
    saison_data = df.groupby(["saison", "langue"])["distress_score"].mean().unstack(fill_value=np.nan)
    saison_data = saison_data.reindex(SAISON_ORDER)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(SAISON_ORDER))
    width = 0.35
    langs = [l for l in ["en", "fr"] if l in saison_data.columns]

    for i, lang in enumerate(langs):
        offset = (i - len(langs)/2 + 0.5) * width
        color = FORUM_COLORS.get(lang, "#888780")
        vals = saison_data[lang].fillna(0).values
        bars = ax.bar(x + offset, vals, width, label=lang, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{val:.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(SAISON_ORDER)
    ax.set_title("Distress moyen par saison académique et langue", fontsize=12)
    ax.set_ylabel("Distress score moyen")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "07_distress_saison.png")


# ── 8. Sentiment × jour de la semaine ────────────────────────────────────────

def fig_sentiment_dayofweek(df):
    if "score_cont" not in df.columns:
        return

    df = df.copy()
    df["date_heure_post"] = pd.to_datetime(df["date_heure_post"], errors="coerce")
    df["jour_num"]        = df["date_heure_post"].dt.dayofweek
    df["jour"]            = df["date_heure_post"].dt.day_name()

    jours_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    jours_fr    = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]

    day_data = df.groupby(["jour_num","langue"])["score_cont"].mean().unstack(fill_value=np.nan)
    day_data = day_data.reindex(range(7))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Patterns hebdomadaires", fontsize=13, y=1.02)

    for lang in ["en", "fr"]:
        if lang in day_data.columns:
            color = FORUM_COLORS.get(lang, "#888780")
            axes[0].plot(jours_fr, day_data[lang].values, marker="o",
                        markersize=5, linewidth=2, color=color, label=lang)
    axes[0].axhline(0, color="#888780", linestyle="--", linewidth=0.8)
    axes[0].set_title("Score sentiment par jour")
    axes[0].set_ylabel("Score moyen")
    axes[0].legend(fontsize=9)

    # Volume de posts par jour
    day_counts = df.groupby("jour_num").size().reindex(range(7), fill_value=0)
    axes[1].bar(jours_fr, day_counts.values, color="#5DCAA5", edgecolor="white")
    axes[1].set_title("Volume de posts par jour")
    axes[1].set_ylabel("Nombre de posts")

    fig.tight_layout()
    save(fig, "08_patterns_hebdomadaires.png")


# ── 9. Top posts haute détresse ───────────────────────────────────────────────

def fig_high_distress_posts(df):
    if "distress_score" not in df.columns:
        return

    top = df.nlargest(10, "distress_score")[
        [c for c in ["forum", "langue", "distress_score", "sentiment_label",
                     "emotion_dominant", "texte_clean"] if c in df.columns]
    ].copy()

    if "texte_clean" in top.columns:
        top["extrait"] = top["texte_clean"].str[:80] + "..."

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    cols_show = [c for c in ["forum", "langue", "distress_score", "sentiment_label",
                              "emotion_dominant", "extrait"] if c in top.columns]
    table = ax.table(
        cellText=top[cols_show].values,
        colLabels=cols_show,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(range(len(cols_show)))
    ax.set_title("Top 10 posts avec distress score le plus élevé", fontsize=12, pad=20)
    fig.tight_layout()
    save(fig, "09_top_distress_posts.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Starting visualization ===")

    # Charger le fichier le plus enrichi
    paths_to_try = [
        Path("data/processed/reddit_topics.csv"),
        Path("data/processed/reddit_emotions.csv"),
        Path("data/processed/reddit_sentiment.csv"),
        Path("data/processed/reddit_clean.csv"),
    ]

    df = None
    for p in paths_to_try:
        if p.exists():
            try:
                df = pd.read_csv(p)
                log.info(f"Loaded {len(df):,} posts from {p}")
                break
            except Exception as e:
                log.warning(f"Could not load {p}: {e}")

    if df is None:
        log.error("No data file found. Run preprocessing first.")
        return

    # Convertir booléens
    for col in ["is_empty", "is_too_short"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(False)

    log.info(f"Columns available: {list(df.columns)}")

    # Générer toutes les figures
    fig_corpus_overview(df)
    fig_sentiment_global(df)
    fig_emotions_global(df)
    fig_distress(df)
    fig_emotions_temporal(df)
    fig_emotions_heatmap(df)
    fig_distress_saison(df)
    fig_sentiment_dayofweek(df)
    fig_high_distress_posts(df)

    log.info(f"=== Done — figures saved to {FIGURES_DIR} ===")


if __name__ == "__main__":
    main()