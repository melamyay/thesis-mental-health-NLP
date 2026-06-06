"""
Dashboard — From Text to Graph
Pipeline NLP pour l'analyse de la santé mentale étudiante sur Reddit
M2 Data Science en Santé — ILIS UFR3S Lille — 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="From Text to Graph — Mental Health NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PALETTE & STYLE
# ─────────────────────────────────────────────
COLORS = {
    "primary":    "#6C63FF",   # violet doux
    "secondary":  "#48CAE4",   # bleu clair
    "accent":     "#F4A261",   # orange chaud
    "danger":     "#E63946",   # rouge distress
    "success":    "#2EC4B6",   # vert teal
    "neutral":    "#A8DADC",   # bleu-gris neutre
    "bg_card":    "#1E1E2E",
    "bg_main":    "#13131F",
    "text":       "#E2E8F0",
    "text_muted": "#94A3B8",
}

EMOTION_COLORS = {
    "anger":        "#E63946",
    "anticipation": "#F4A261",
    "disgust":      "#6D4C41",
    "fear":         "#9B2335",
    "joy":          "#2EC4B6",
    "sadness":      "#6C63FF",
    "surprise":     "#FFD166",
    "trust":        "#1A535C",
}

SENTIMENT_COLORS = {
    "negative": "#E63946",
    "neutral":  "#94A3B8",
    "positive": "#2EC4B6",
}

SUBREDDIT_COLORS = {
    "r/Students":    "#6C63FF",
    "r/etudiants":   "#48CAE4",
    "r/csMajors":    "#F4A261",
    "r/CollegeRant": "#E63946",
    "r/Student":     "#2EC4B6",
    "r/college":     "#FFD166",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* Force dark mode everywhere */
:root { color-scheme: dark !important; }

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #13131F !important;
    color: #E2E8F0 !important;
}

[data-testid="stAppViewContainer"] > .main { background-color: #13131F !important; }
[data-testid="stHeader"] { background-color: #13131F !important; }
.stSelectbox > div > div, .stMultiSelect > div > div,
.stTextInput > div > div, .stNumberInput > div > div {
    background-color: #1E1E2E !important; color: #E2E8F0 !important; border-color: #2D2D44 !important;
}
.stRadio > label, .stCheckbox > label, label { color: #E2E8F0 !important; }
.stMarkdown, .stText, p { color: #E2E8F0 !important; }
[data-baseweb="select"] { background-color: #1E1E2E !important; }
[data-baseweb="popover"] { background-color: #1E1E2E !important; }
[data-testid="baseButton-headerNoPadding"] { display: none !important; }

.main { background-color: #13131F; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid #2D2D44;
}
[data-testid="stSidebar"] .stRadio > label { color: #E2E8F0 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
    border: 1px solid #2D2D44;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #6C63FF;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #6C63FF;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.78rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-sub {
    font-size: 0.72rem;
    color: #6C63FF;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6C63FF;
    margin-bottom: 0.3rem;
}
.page-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    color: #94A3B8;
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
}

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #1E1E2E, #252540);
    border-left: 3px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
    color: #CBD5E1;
}
.insight-box.warning { border-left-color: #F4A261; }
.insight-box.danger  { border-left-color: #E63946; }
.insight-box.success { border-left-color: #2EC4B6; }

/* Tags */
.tag {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #6C63FF;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}

/* Nav pill */
.nav-pill {
    background: rgba(108,99,255,0.15);
    border: 1px solid #6C63FF;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    color: #6C63FF;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

/* Stickers */
.badge {
    background: rgba(46,196,182,0.15);
    border: 1px solid #2EC4B6;
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.72rem;
    color: #2EC4B6;
    font-family: 'Space Mono', monospace;
}

/* Table */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

hr { border-color: #2D2D44; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Charge les CSV depuis data/processed/ avec fallback sur données simulées."""
    base = Path("data/processed")

    def try_load(filename):
        p = base / filename
        if p.exists():
            return pd.read_csv(p)
        return None

    df_topics   = try_load("reddit_topics.csv")
    df_emotions = try_load("reddit_emotions.csv")
    df_sentiment= try_load("reddit_sentiment.csv")
    df_clean    = try_load("reddit_clean.csv")

    # Priorité : fichier le plus enrichi d'abord
    df = None
    for candidate in [df_topics, df_emotions, df_sentiment, df_clean]:
        if candidate is not None:
            df = candidate
            break

    if df is None:
        # ── Données de démo basées sur les vrais résultats ──
        np.random.seed(42)
        n = 10412
        subreddits = ["Students","etudiants","csMajors","CollegeRant","Student","college"]
        sub_counts = [1998, 1991, 1990, 1886, 1780, 767]
        sub_list   = np.repeat(subreddits, sub_counts)
        np.random.shuffle(sub_list)

        months_dist = {8:155, 9:193, 10:120, 11:130, 12:115,
                       1:566, 2:1260, 3:2680, 4:893, 5:3050,
                       6:175, 7:75}
        month_vals = np.concatenate([np.full(v, k) for k,v in months_dist.items()])
        np.random.shuffle(month_vals)

        lang = np.random.choice(["en","fr","other"],n,p=[0.795,0.186,0.019])

        sentiment_dist = np.random.choice(["negative","neutral","positive"],n,p=[0.375,0.410,0.215])
        score_cont = np.where(
            sentiment_dist=="negative", np.random.normal(-0.25,0.3,n),
            np.where(sentiment_dist=="positive", np.random.normal(0.3,0.2,n),
                     np.random.normal(0.01,0.15,n))
        ).clip(-1,1)

        emotions = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
        emotion_dom = np.random.choice(emotions,n,
            p=[0.094,0.372,0.0003,0.032,0.040,0.136,0.133,0.200])

        nas_sim = np.random.beta(1.2, 2.5, n)
        nas_level_sim = pd.cut(nas_sim,
                               bins=[-0.001,
                                     np.percentile(nas_sim, 33.3),
                                     np.percentile(nas_sim, 66.6), 1.0],
                               labels=["low","moderate","high"])

        days_of_week = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        dayofweek = np.random.choice(days_of_week, n,
                        p=[0.151,0.154,0.152,0.161,0.136,0.115,0.131])

        texts_sample = [
            "I feel I am just paying college to be miserable lonely and depressed...",
            "I lost my boyfriend and my best friend. I feel so lonely that it's killing me.",
            "class 11th - severe anxiety and chances of failing...",
            "International student struggling to finish final semester...",
            "I am feeling lost as a senior in college and compsci major...",
        ]

        df = pd.DataFrame({
            "forum": sub_list,
            "lang": lang,
            "month": month_vals,
            "dayofweek": dayofweek,
            "sentiment_label": sentiment_dist,
            "score_cont": score_cont,
            "emotion_dominant": emotion_dom,
            "nas_score": nas_sim,
            "nas_level": nas_level_sim,
            "text_clean": np.random.choice(texts_sample, n),
        })

        # Saisons académiques
        season_map = {8:"ete",9:"rentree",10:"automne",11:"automne",12:"automne",
                      1:"hiver",2:"hiver",3:"hiver",4:"printemps",5:"examens",
                      6:"examens",7:"ete"}
        df["saison"] = df["month"].map(season_map)

        # Date fictive
        year_map  = {8:2025,9:2025,10:2025,11:2025,12:2025,
                     1:2026,2:2026,3:2026,4:2026,5:2026,6:2026,7:2025}
        df["year"] = df["month"].map(year_map)
        df["date"] = pd.to_datetime(
            df.apply(lambda r: f"{r['year']}-{int(r['month']):02d}-15", axis=1)
        )

    else:
        # Normalisation colonnes
        if "subreddit" in df.columns and "forum" not in df.columns:
            df.rename(columns={"subreddit":"forum"}, inplace=True)
        if "langue" in df.columns and "lang" not in df.columns:
            df.rename(columns={"langue":"lang"}, inplace=True)

        for col in ["date_heure_post", "date", "created_utc"]:
            if col in df.columns:
                try:
                    df["date"] = pd.to_datetime(df[col], unit="s" if "utc" in col else None)
                    df["month"] = df["date"].dt.month
                    df["year"]  = df["date"].dt.year
                    break
                    if "mois_post" in df.columns and "month" not in df.columns:
                        df["month"] = df["mois_post"]
                    if "annee_post" in df.columns and "year" not in df.columns:
                        df["year"] = df["annee_post"]
                except: pass

        if "month" not in df.columns and "date" in df.columns:
            df["month"] = df["date"].dt.month

    return df

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.2rem;'>🧠</div>
        <div style='font-family:Space Mono,monospace; font-size:0.78rem; color:#6C63FF; font-weight:700; line-height:1.35;'>
            FROM TEXT TO GRAPH
        </div>
        <div style='font-size:0.65rem; color:#94A3B8; margin-top:0.35rem; line-height:1.5; padding:0 0.3rem;'>
            A Reproducible NLP Pipeline for the Longitudinal Analysis of Student Mental Health on Reddit
        </div>
        <div style='font-size:0.65rem; color:#64748B; margin-top:0.3rem;'>
            M2 DSS — ILIS UFR3S Lille — 2026
        </div>
    </div>
    <hr style='border-color:#2D2D44; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊  Vue d'ensemble",
         "💬  Sentiment",
         "🎭  Émotions",
         "🌡️  Distress",
         "📈  Patterns temporels",
         "🕸️  Graphes & Topics"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2D2D44;'>", unsafe_allow_html=True)

    # Filtres globaux
    st.markdown("<div class='section-title'>Filtres</div>", unsafe_allow_html=True)
    df_raw = load_data()

    subreddits_all = sorted(df_raw["forum"].unique().tolist()) if "forum" in df_raw.columns else []
    selected_subs = st.multiselect(
        "Subreddits",
        options=subreddits_all,
        default=subreddits_all,
        label_visibility="visible"
    )

    if "lang" in df_raw.columns:
        langs_all = [l for l in df_raw["lang"].unique() if l in ["en","fr"]]
    else:
        langs_all = ["en","fr"]
    selected_langs = langs_all  # Pas de filtre langue — toutes les langues affichées

    st.markdown("<hr style='border-color:#2D2D44;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#475569; text-align:center; line-height:1.6;'>
        Corpus : août 2025 → mai 2026<br>
        10 412 posts · 6 subreddits<br>
        EN 79.5% · FR 18.6%
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTRAGE
# ─────────────────────────────────────────────
df = df_raw.copy()
if selected_subs and "forum" in df.columns:
    df = df[df["forum"].isin(selected_subs)]
if selected_langs and "lang" in df.columns:
    df = df[df["lang"].isin(selected_langs)]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def styled_plotly(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,46,0.6)",
        font=dict(family="DM Sans", color="#CBD5E1", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(
            bgcolor="rgba(30,30,46,0.8)",
            bordercolor="#2D2D44",
            borderwidth=1,
        ),
        xaxis=dict(gridcolor="#2D2D44", zerolinecolor="#2D2D44"),
        yaxis=dict(gridcolor="#2D2D44", zerolinecolor="#2D2D44"),
    )
    return fig

def metric_card(value, label, sub=""):
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return f"""
    <div class='metric-card'>
        <div class='metric-value'>{value}</div>
        <div class='metric-label'>{label}</div>
        {sub_html}
    </div>"""

def insight(text, kind=""):
    cls = f"insight-box {kind}".strip()
    return f"<div class='{cls}'>{text}</div>"

# ─────────────────────────────────────────────
# PAGE 1 — VUE D'ENSEMBLE
# ─────────────────────────────────────────────
if page == "📊  Vue d'ensemble":
    st.markdown("<div class='section-title'>Corpus Reddit</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Vue d'ensemble du corpus</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Données collectées sur 6 subreddits · Août 2025 → Mai 2026</div>", unsafe_allow_html=True)

    # ── KPIs ──
    n_posts    = len(df)
    n_subs     = df["forum"].nunique() if "forum" in df.columns else 6
    n_en       = len(df[df["lang"]=="en"]) if "lang" in df.columns else 8275
    n_fr       = len(df[df["lang"]=="fr"]) if "lang" in df.columns else 1939

    if "date" in df.columns:
        date_min = df["date"].min().strftime("%d %b %Y")
        date_max = df["date"].max().strftime("%d %b %Y")
    else:
        date_min, date_max = "Août 2025", "Mars 2026"

    cols = st.columns(5)
    cards = [
        (f"{n_posts:,}", "Posts analysés", "après déduplication"),
        (str(n_subs), "Subreddits", "EN & FR"),
        (f"{n_en:,}", "Posts EN", f"{n_en/max(n_posts,1)*100:.1f}%"),
        (f"{n_fr:,}", "Posts FR", f"{n_fr/max(n_posts,1)*100:.1f}%"),
        ("806", "Doublons retirés", "entre les 2 batches"),
    ]
    for col, (v, l, s) in zip(cols, cards):
        col.markdown(metric_card(v, l, s), unsafe_allow_html=True)

    st.markdown(f"""
    <div style='display:flex; gap:0.5rem; margin:1rem 0; flex-wrap:wrap; align-items:center;'>
        <span style='font-size:0.8rem; color:#64748B;'>Période :</span>
        <span class='badge'>{date_min}</span>
        <span style='color:#64748B;'>→</span>
        <span class='badge'>{date_max}</span>
        <span style='margin-left:1rem; font-size:0.8rem; color:#64748B;'>Scraping :</span>
        <span class='badge'>2 batches</span>
        <span class='badge'>287 conflits de langue corrigés</span>
        <span class='badge'>1 179 posts courts flaggés</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Graphiques ──
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("**Posts par subreddit**")
        sub_counts_real = {
            "r/Students":1998,"r/etudiants":1991,"r/csMajors":1990,
            "r/CollegeRant":1886,"r/Student":1780,"r/college":767
        }
        if "forum" in df.columns:
            vc = df["forum"].value_counts().reset_index()
            vc.columns = ["subreddit","count"]
            vc["subreddit"] = vc["subreddit"].apply(lambda x: f"r/{x}" if not x.startswith("r/") else x)
        else:
            vc = pd.DataFrame({"subreddit":list(sub_counts_real.keys()),
                                "count":list(sub_counts_real.values())})

        fig = px.bar(
            vc.sort_values("count"),
            x="count", y="subreddit", orientation="h",
            color="subreddit",
            color_discrete_map={k: list(SUBREDDIT_COLORS.values())[i]
                                 for i,k in enumerate(vc["subreddit"])},
            text="count",
        )
        fig.update_traces(textposition="outside", textfont_color="#CBD5E1",
                          marker_line_width=0, width=0.65)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(styled_plotly(fig, 340), use_container_width=True)

    with col2:
        st.markdown("**Distribution des langues**")
        lang_data = {"EN":n_en, "FR":n_fr, "Autre":n_posts-n_en-n_fr}
        fig2 = go.Figure(go.Pie(
            labels=list(lang_data.keys()),
            values=list(lang_data.values()),
            hole=0.55,
            marker_colors=["#6C63FF","#2EC4B6","#475569"],
            textinfo="label+percent",
            textfont_color="#E2E8F0",
        ))
        fig2.update_layout(
            annotations=[dict(text=f"<b>{n_posts:,}</b><br>posts", x=0.5, y=0.5,
                              font_size=14, font_color="#E2E8F0", showarrow=False)],
            showlegend=False,
        )
        st.plotly_chart(styled_plotly(fig2, 340), use_container_width=True)

    # Volume mensuel
    st.markdown("**Volume de posts par mois**")
    if "month" in df.columns:
        month_counts = df.groupby(["month","lang"]).size().reset_index(name="count")
        month_name = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",
                      6:"Juin",7:"Juil",8:"Août",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
        month_counts["mois"] = month_counts["month"].map(month_name)

        fig3 = px.bar(
            month_counts, x="mois", y="count", color="lang",
            color_discrete_map={"en":"#6C63FF","fr":"#2EC4B6","other":"#475569"},
            barmode="stack",
            category_orders={"mois":["Août","Sep","Oct","Nov","Déc","Jan","Fév","Mar","Avr","Mai","Juin","Juil"]},
            labels={"count":"Nombre de posts","mois":"","lang":"Langue"},
        )
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(styled_plotly(fig3, 280), use_container_width=True)

    # Insights
    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.5rem;'>
    """ +
    insight("📌 <b>r/etudiants</b> et <b>r/Students</b> sont les subreddits les plus actifs (~1 990 posts chacun).") +
    insight("📅 <b>Pic en mai 2026</b> — période d'examens — avec 3 050 posts.", "warning") +
    insight("🌍 <b>79.5% des posts en anglais</b>, malgré l'inclusion de subreddits francophones.", "success") +
    insight("🔎 <b>2 batches de scraping</b> fusionnés, 806 doublons supprimés par déduplication sur <code>id_post</code>.") +
    "</div>", unsafe_allow_html=True)

    # ── Explorateur de posts ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Exploration du corpus</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title' style='font-size:1.3rem;'>Posts exemples</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Parcourez des posts réels filtrés par émotion dominante et niveau NAS</div>", unsafe_allow_html=True)

    df_posts = None
    posts_base = Path("data/processed")
    emo_path = posts_base / "reddit_emotions.csv"
    if emo_path.exists():
        df_posts = pd.read_csv(emo_path)
    elif Path("reddit_emotions.csv").exists():
        df_posts = pd.read_csv("reddit_emotions.csv")

    if df_posts is not None:
        df_posts = df_posts[
            (~df_posts["is_too_short"]) &
            (df_posts["texte_clean"].notna()) &
            (df_posts["texte_clean"].str.len() > 20) &
            (df_posts["nas_score"].notna()) &
            (df_posts["emotion_dominant"].notna())
        ].copy()

        col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 0.8])
        with col_f1:
            emotions_available = sorted(df_posts["emotion_dominant"].dropna().unique().tolist())
            selected_emotion = st.selectbox("🎭 Émotion dominante", ["Toutes"] + emotions_available, key="explorer_emotion")
        with col_f2:
            nas_labels = {"low": "🟢 Low", "moderate": "🟡 Moderate", "high": "🔴 High"}
            selected_nas = st.selectbox("📊 Niveau NAS", ["Tous"] + list(nas_labels.values()), key="explorer_nas")
        with col_f3:
            n_display = st.slider("Nb posts", min_value=3, max_value=15, value=5, key="explorer_n")

        df_filtered = df_posts.copy()
        if selected_emotion != "Toutes":
            df_filtered = df_filtered[df_filtered["emotion_dominant"] == selected_emotion]
        if selected_nas != "Tous":
            nas_reverse = {v: k for k, v in nas_labels.items()}
            df_filtered = df_filtered[df_filtered["nas_level"] == nas_reverse[selected_nas]]

        if len(df_filtered) == 0:
            st.markdown("<div style='background:rgba(108,99,255,0.08); border:1px dashed #6C63FF; border-radius:12px; padding:1.5rem; text-align:center; color:#94A3B8;'>Aucun post ne correspond à ces filtres.</div>", unsafe_allow_html=True)
        else:
            sample = df_filtered.sample(n=min(n_display, len(df_filtered)), random_state=42)
            st.markdown(f"<div style='font-size:0.78rem; color:#64748B; margin-bottom:0.8rem;'>{len(df_filtered):,} posts correspondent · affichage de {len(sample)}</div>", unsafe_allow_html=True)

            for _, row in sample.iterrows():
                emo_color = EMOTION_COLORS.get(row["emotion_dominant"], "#6C63FF")
                nas_color = {"low":"#2EC4B6","moderate":"#F4A261","high":"#E63946"}.get(str(row.get("nas_level","")).lower(), "#94A3B8")
                sentiment_color = {"negative":"#E63946","neutral":"#94A3B8","positive":"#2EC4B6"}.get(str(row.get("sentiment_label","")).lower(), "#94A3B8")
                forum_val = str(row.get("forum",""))
                if not forum_val.startswith("r/"): forum_val = f"r/{forum_val}"
                langue_val = str(row.get("langue","")).upper()
                nas_val = float(row.get("nas_score", 0))
                emo_val = str(row.get("emotion_dominant","")).capitalize()
                sent_val = str(row.get("sentiment_label","")).capitalize()
                titre_val = str(row.get("titre","")) if pd.notna(row.get("titre")) else ""
                texte_val = str(row.get("texte_clean",""))
                extrait = texte_val[:280] + "…" if len(texte_val) > 280 else texte_val
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#1E1E2E,#252540); border:1px solid #2D2D44; border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.7rem; border-left:3px solid {emo_color};'>
                    <div style='display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.55rem; align-items:center;'>
                        <span style='font-family:Space Mono,monospace; font-size:0.72rem; color:#6C63FF; background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.3); border-radius:6px; padding:2px 8px;'>{forum_val}</span>
                        <span style='font-size:0.72rem; color:#64748B; background:#1a1a2e; border-radius:6px; padding:2px 8px;'>{langue_val}</span>
                        <span style='font-size:0.72rem; color:{emo_color}; background:rgba(0,0,0,0.2); border:1px solid {emo_color}40; border-radius:6px; padding:2px 8px;'>🎭 {emo_val}</span>
                        <span style='font-size:0.72rem; color:{nas_color}; background:rgba(0,0,0,0.2); border:1px solid {nas_color}40; border-radius:6px; padding:2px 8px;'>NAS {nas_val:.3f}</span>
                        <span style='font-size:0.72rem; color:{sentiment_color}; background:rgba(0,0,0,0.2); border:1px solid {sentiment_color}40; border-radius:6px; padding:2px 8px;'>{sent_val}</span>
                    </div>
                    {f"<div style='font-size:0.85rem; font-weight:600; color:#E2E8F0; margin-bottom:0.35rem;'>{titre_val}</div>" if titre_val else ""}
                    <div style='font-size:0.82rem; color:#94A3B8; line-height:1.55;'>{extrait}</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🔀 Nouveaux exemples", key="resample_btn"):
                st.rerun()
    else:
        fallback_posts = [
            {"forum":"CollegeRant","langue":"EN","emotion_dominant":"anger","nas_score":0.9739,"nas_level":"high","sentiment_label":"negative","titre":"Frustration in the dorm","texte_clean":"Everyone in this dorm cannot stand you and wishes you the worst. You are the rudest most inconsiderate person imaginable..."},
            {"forum":"CollegeRant","langue":"EN","emotion_dominant":"sadness","nas_score":0.9727,"nas_level":"high","sentiment_label":"negative","titre":"Suicidal ideation & academic pressure","texte_clean":"I have already been suicidal for more than four years and it has become way too overwhelming for somebody with severe AD..."},
            {"forum":"csMajors","langue":"EN","emotion_dominant":"sadness","nas_score":0.9727,"nas_level":"high","sentiment_label":"negative","titre":"Feeling lost as a senior","texte_clean":"I am feeling lost as a senior in college and compsci major. Nothing feels like it's going the way I planned..."},
            {"forum":"Students","langue":"EN","emotion_dominant":"anticipation","nas_score":0.220,"nas_level":"moderate","sentiment_label":"neutral","titre":"Preparing for internship interviews","texte_clean":"I have three interviews lined up this week and I genuinely don't know if I'm ready..."},
            {"forum":"etudiants","langue":"FR","emotion_dominant":"trust","nas_score":0.105,"nas_level":"low","sentiment_label":"neutral","titre":"Conseils pour trouver un stage","texte_clean":"Bonjour, je cherche des conseils pour trouver un stage en data science..."},
        ]
        st.markdown("<div style='background:rgba(244,162,97,0.08); border:1px solid rgba(244,162,97,0.3); border-radius:10px; padding:0.6rem 1rem; margin-bottom:1rem; font-size:0.78rem; color:#F4A261;'>⚠️ Mode démo — données simulées. Placez <code>reddit_emotions.csv</code> dans <code>data/processed/</code> pour afficher les vrais posts.</div>", unsafe_allow_html=True)
        for post in fallback_posts:
            emo_color = EMOTION_COLORS.get(post["emotion_dominant"], "#6C63FF")
            nas_color = {"low":"#2EC4B6","moderate":"#F4A261","high":"#E63946"}.get(post["nas_level"],"#94A3B8")
            sentiment_color = {"negative":"#E63946","neutral":"#94A3B8","positive":"#2EC4B6"}.get(post["sentiment_label"],"#94A3B8")
            forum_val = post["forum"] if post["forum"].startswith("r/") else f"r/{post['forum']}"
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1E1E2E,#252540); border:1px solid #2D2D44; border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.7rem; border-left:3px solid {emo_color};'>
                <div style='display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.55rem; align-items:center;'>
                    <span style='font-family:Space Mono,monospace; font-size:0.72rem; color:#6C63FF; background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.3); border-radius:6px; padding:2px 8px;'>{forum_val}</span>
                    <span style='font-size:0.72rem; color:#64748B; background:#1a1a2e; border-radius:6px; padding:2px 8px;'>{post["langue"]}</span>
                    <span style='font-size:0.72rem; color:{emo_color}; background:rgba(0,0,0,0.2); border:1px solid {emo_color}40; border-radius:6px; padding:2px 8px;'>🎭 {post["emotion_dominant"].capitalize()}</span>
                    <span style='font-size:0.72rem; color:{nas_color}; background:rgba(0,0,0,0.2); border:1px solid {nas_color}40; border-radius:6px; padding:2px 8px;'>NAS {post["nas_score"]:.3f}</span>
                    <span style='font-size:0.72rem; color:{sentiment_color}; background:rgba(0,0,0,0.2); border:1px solid {sentiment_color}40; border-radius:6px; padding:2px 8px;'>{post["sentiment_label"].capitalize()}</span>
                </div>
                <div style='font-size:0.85rem; font-weight:600; color:#E2E8F0; margin-bottom:0.35rem;'>{post["titre"]}</div>
                <div style='font-size:0.82rem; color:#94A3B8; line-height:1.55;'>{post["texte_clean"]}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — SENTIMENT
# ─────────────────────────────────────────────
elif page == "💬  Sentiment":
    st.markdown("<div class='section-title'>Analyse NLP</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Analyse de sentiment</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>CamemBERT (FR) · RoBERTa (EN) · Score continu [-1, 1]</div>", unsafe_allow_html=True)

    # KPIs
    if "sentiment_label" in df.columns:
        neg = (df["sentiment_label"]=="negative").mean()*100
        neu = (df["sentiment_label"]=="neutral").mean()*100
        pos = (df["sentiment_label"]=="positive").mean()*100
    else:
        neg, neu, pos = 37.5, 41.0, 21.5

    score_mean = df["score_cont"].mean() if "score_cont" in df.columns else -0.111
    score_en = df.loc[df["lang"] == "en", "score_cont"].mean() if (
                "score_cont" in df.columns and "lang" in df.columns) else -0.117
    score_fr = df.loc[df["lang"] == "fr", "score_cont"].mean() if (
                "score_cont" in df.columns and "lang" in df.columns) else -0.088

    cols = st.columns(5)
    for col, (v,l,s) in zip(cols, [
        (f"{neg:.1f}%","Négatif","37.5% corpus complet"),
        (f"{neu:.1f}%","Neutre","41% corpus complet"),
        (f"{pos:.1f}%","Positif","21.5% corpus complet"),
        (f"{score_en:.2f}","Score moyen EN","[-1, 1]"),
        (f"{score_fr:.2f}","Score moyen FR","[-1, 1]"),
    ]):
        col.markdown(metric_card(v,l,s), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution globale**")
        labels = ["Négatif","Neutre","Positif"]
        values = [neg, neu, pos]
        colors = ["#E63946","#94A3B8","#2EC4B6"]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.5,
            marker_colors=colors,
            textinfo="label+percent",
            textfont_color="#E2E8F0",
        ))
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_plotly(fig, 320), use_container_width=True)

    with col2:
        st.markdown("**Distribution du score continu par langue**")
        if "score_cont" in df.columns and "lang" in df.columns:
            fig2 = go.Figure()
            for lang, color in [("en","#6C63FF"),("fr","#2EC4B6")]:
                sub = df[df["lang"]==lang]["score_cont"].dropna()
                if len(sub):
                    fig2.add_trace(go.Histogram(
                        x=sub, name=lang.upper(),
                        marker_color=color, opacity=0.75,
                        nbinsx=40,
                    ))
            fig2.add_vline(x=0, line_dash="dash", line_color="#F4A261",
                           annotation_text="Neutre", annotation_font_color="#F4A261")
            fig2.update_layout(barmode="overlay", xaxis_title="Score [-1, 1]", yaxis_title="Nb posts")
        else:
            # Données simulées
            x_en = np.random.normal(-0.18, 0.35, 7389)
            x_fr = np.random.normal(-0.10, 0.30, 1844)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=x_en, name="EN", marker_color="#6C63FF", opacity=0.75, nbinsx=40))
            fig2.add_trace(go.Histogram(x=x_fr, name="FR", marker_color="#2EC4B6", opacity=0.75, nbinsx=40))
            fig2.add_vline(x=0, line_dash="dash", line_color="#F4A261")
            fig2.update_layout(barmode="overlay", xaxis_title="Score [-1, 1]", yaxis_title="Nb posts")
        st.plotly_chart(styled_plotly(fig2, 320), use_container_width=True)

    # Sentiment par subreddit
    st.markdown("**Sentiment par subreddit**")
    if "sentiment_label" in df.columns and "forum" in df.columns:
        sub_sent = df.groupby(["forum","sentiment_label"]).size().reset_index(name="count")
        sub_total = df.groupby("forum").size().reset_index(name="total")
        sub_sent = sub_sent.merge(sub_total, on="forum")
        sub_sent["pct"] = sub_sent["count"] / sub_sent["total"] * 100
        fig3 = px.bar(
            sub_sent, x="forum", y="pct", color="sentiment_label",
            color_discrete_map={"negative":"#E63946","neutral":"#94A3B8","positive":"#2EC4B6"},
            labels={"pct":"% posts","forum":"Subreddit","sentiment_label":"Sentiment"},
            barmode="stack",
        )
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(yaxis_title="%", xaxis_title="")
        st.plotly_chart(styled_plotly(fig3, 300), use_container_width=True)

    # Score mensuel
    st.markdown("**Score sentiment moyen par mois**")
    month_name = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",
                  6:"Juin",7:"Juil",8:"Août",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
    if "score_cont" in df.columns and "month" in df.columns:
        monthly = df.groupby(["month","lang"])["score_cont"].mean().reset_index()
        monthly["mois"] = monthly["month"].map(month_name)
        fig4 = px.line(
            monthly[monthly["lang"].isin(["en","fr"])],
            x="mois", y="score_cont", color="lang",
            color_discrete_map={"en":"#6C63FF","fr":"#2EC4B6"},
            markers=True,
            category_orders={"mois":["Août","Sep","Oct","Nov","Déc","Jan","Fév","Mar","Avr","Mai"]},
        )
        fig4.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
        fig4.update_layout(yaxis_title="Score moyen", xaxis_title="")
        st.plotly_chart(styled_plotly(fig4, 280), use_container_width=True)

    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.5rem;'>
    """ +
    insight("📉 <b>37.5% des posts sont négatifs</b> — le corpus reflète clairement un contexte de détresse étudiante.", "danger") +
    insight("🇬🇧 <b>EN plus négatif (−0.117)</b> que FR (−0.088), probablement dû aux subreddits CollegeRant et csMajors.", "warning") +
    insight("📅 Score plus négatif <b>en août et en période d'examens</b>.", "warning") +
    insight("🔬 Score continu [-1,1] combinant scores positif, négatif et neutre des modèles BERT.", "success") +
    "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 — ÉMOTIONS
# ─────────────────────────────────────────────
elif page == "🎭  Émotions":
    st.markdown("<div class='section-title'>Analyse NLP</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Analyse des émotions</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>NRC Lexicon (30%) + go-emotions BERT roberta-base (70%) · 8 émotions NRC</div>", unsafe_allow_html=True)

    # KPIs émotions
    emotion_dominant_counts = {
        "anticipation":3412,"trust":1834,"sadness":1247,
        "surprise":1216,"anger":862,"joy":364,"fear":295,"disgust":3
    }
    top_emotion = max(emotion_dominant_counts, key=emotion_dominant_counts.get)

    cols = st.columns(4)
    for col, (v,l,s) in zip(cols, [
        (top_emotion.capitalize(),"Émotion dominante","corpus complet"),
        ("3 412","Posts — anticipation","32.8% du corpus"),
        ("0.219","Score anticipation","score moyen max"),
        ("8","Émotions NRC","classification finale"),
    ]):
        col.markdown(metric_card(v,l,s), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Score moyen par émotion**")
        emotion_scores = {
            "anger":0.091,"anticipation":0.219,"disgust":0.009,
            "fear":0.029,"joy":0.055,"sadness":0.095,
            "surprise":0.142,"trust":0.134,
        }
        df_emo = pd.DataFrame({"emotion":list(emotion_scores.keys()),
                                "score":list(emotion_scores.values())})
        fig = px.bar(
            df_emo, x="emotion", y="score",
            color="emotion",
            color_discrete_map=EMOTION_COLORS,
            text="score",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                          textfont_color="#CBD5E1", marker_line_width=0, width=0.7)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Score moyen")
        st.plotly_chart(styled_plotly(fig, 320), use_container_width=True)

    with col2:
        st.markdown("**Émotion dominante par post (nb posts)**")
        df_dom = pd.DataFrame({
            "emotion": list(emotion_dominant_counts.keys()),
            "count":   list(emotion_dominant_counts.values()),
        }).sort_values("count")
        fig2 = px.bar(
            df_dom, x="count", y="emotion", orientation="h",
            color="emotion",
            color_discrete_map=EMOTION_COLORS,
            text="count",
        )
        fig2.update_traces(textposition="outside", textfont_color="#CBD5E1",
                           marker_line_width=0, width=0.65)
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(styled_plotly(fig2, 320), use_container_width=True)

    # Heatmap subreddit × émotion
    st.markdown("**Score moyen par émotion et subreddit (heatmap)**")
    heatmap_data = {
        "CollegeRant": {"anger":0.178,"anticipation":0.113,"disgust":0.014,"fear":0.058,"joy":0.051,"sadness":0.213,"surprise":0.111,"trust":0.071},
        "Student":     {"anger":0.060,"anticipation":0.243,"disgust":0.007,"fear":0.026,"joy":0.073,"sadness":0.065,"surprise":0.113,"trust":0.197},
        "Students":    {"anger":0.066,"anticipation":0.231,"disgust":0.008,"fear":0.022,"joy":0.068,"sadness":0.059,"surprise":0.114,"trust":0.218},
        "college":     {"anger":0.077,"anticipation":0.220,"disgust":0.008,"fear":0.053,"joy":0.057,"sadness":0.132,"surprise":0.165,"trust":0.092},
        "csMajors":    {"anger":0.052,"anticipation":0.311,"disgust":0.005,"fear":0.022,"joy":0.057,"sadness":0.070,"surprise":0.150,"trust":0.108},
        "etudiants":   {"anger":0.096,"anticipation":0.205,"disgust":0.008,"fear":0.009,"joy":0.032,"sadness":0.040,"surprise":0.200,"trust":0.126},
    }
    hm_df = pd.DataFrame(heatmap_data).T
    fig3 = px.imshow(
        hm_df,
        color_continuous_scale=[[0,"#1A535C"],[0.4,"#F0E68C"],[1,"#9B2335"]],
        text_auto=".3f",
        aspect="auto",
        labels={"color":"Score"},
    )
    fig3.update_traces(textfont_color="#1a1a1a")
    fig3.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(styled_plotly(fig3, 320), use_container_width=True)

    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.5rem;'>
    """ +
    insight("🔮 <b>Anticipation dominante</b> (0.219 en score, 3 412 posts) — stress orienté vers l'avenir académique.") +
    insight("😤 <b>csMajors : anticipation maximale (0.311)</b> — stress de carrière tech particulièrement marqué.", "warning") +
    insight("😢 <b>CollegeRant : sadness la plus élevée (0.213)</b> — communauté d'expression de détresse directe.", "danger") +
    insight("🤝 Mix NRC + BERT pondéré (30/70) pour combiner couverture lexicale et compréhension contextuelle.", "success") +
    "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 4 — NAS
# ─────────────────────────────────────────────
elif page == "🌡️  Distress":
    st.markdown("<div class='section-title'>Analyse NLP</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Negative Affect Score (NAS)</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>NAS = NA / (NA + PA + ε) · Fondé sur le modèle bidimensionnel PANAS · 3 niveaux par terciles empiriques</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (v,l,s) in zip(cols, [
        ("0.339","NAS moyen","corpus complet"),
        ("3 065","Posts low","1er tercile"),
        ("2 965","Posts moderate","2e tercile"),
        ("3 203","Posts high","3e tercile — 34.7%"),
    ]):
        col.markdown(metric_card(v,l,s), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution du NAS**")
        if "nas_score" in df.columns:
            nas_data = df["nas_score"].dropna()
        else:
            nas_data = np.random.beta(1.2, 2.5, 9233)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=nas_data, nbinsx=50, name="NAS",
            marker_color="#E63946", opacity=0.8,
        ))
        mean_val = float(np.mean(nas_data))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#F4A261",
                      annotation_text=f"Moy. {mean_val:.3f}",
                      annotation_font_color="#F4A261")
        fig.update_layout(xaxis_title="NAS [0, 1]", yaxis_title="Nb posts", showlegend=False)
        st.plotly_chart(styled_plotly(fig, 300), use_container_width=True)

    with col2:
        st.markdown("**Niveaux NAS (terciles empiriques)**")
        if "nas_level" in df.columns:
            level_counts = df["nas_level"].value_counts().reindex(["low","moderate","high"], fill_value=0).to_dict()
        else:
            level_counts = {"low":3065,"moderate":2965,"high":3203}
        fig2 = px.bar(
            x=list(level_counts.keys()),
            y=list(level_counts.values()),
            color=list(level_counts.keys()),
            color_discrete_map={"low":"#2EC4B6","moderate":"#F4A261","high":"#E63946"},
            text=list(level_counts.values()),
        )
        fig2.update_traces(textposition="outside", textfont_color="#CBD5E1", marker_line_width=0)
        fig2.update_layout(showlegend=False, xaxis_title="Niveau", yaxis_title="Nb posts")
        st.plotly_chart(styled_plotly(fig2, 300), use_container_width=True)

    # NAS par saison
    st.markdown("**NAS moyen par saison académique et langue**")
    saison_data = {
        "saison":  ["rentree","automne","hiver","hiver","printemps","printemps","examens","examens"],
        "langue":  ["en","en","en","fr","en","fr","en","fr"],
        "nas":     [0.481,0.430,0.366,0.269,0.341,0.274,0.348,0.279],
    }
    df_saison = pd.DataFrame(saison_data)
    fig3 = px.bar(
        df_saison, x="saison", y="nas", color="langue", barmode="group",
        color_discrete_map={"en":"#6C63FF","fr":"#2EC4B6"},
        text="nas",
        category_orders={"saison":["rentree","automne","hiver","printemps","examens"]},
        labels={"nas":"NAS moyen","saison":"Saison académique","langue":"Langue"},
    )
    fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                       textfont_color="#CBD5E1", marker_line_width=0)
    fig3.update_layout(yaxis_title="NAS moyen", xaxis_title="")
    st.plotly_chart(styled_plotly(fig3, 300), use_container_width=True)

    # Top NAS posts
    st.markdown("**Top 10 posts — NAS le plus élevé (posts valides)**")
    top_posts = pd.DataFrame({
        "Subreddit":  ["csMajors","Students","CollegeRant","CollegeRant","csMajors",
                       "CollegeRant","Student","CollegeRant","CollegeRant","college"],
        "Langue":     ["en"]*10,
        "NAS":        [0.9778,0.9776,0.9739,0.9727,0.9727,0.9712,0.9710,0.9693,0.9676,0.9671],
        "Sentiment":  ["negative"]*10,
        "Émotion":    ["anger","anger","anger","sadness","sadness","sadness","sadness","anger","anger","sadness"],
        "Extrait":    [
            "The workforce does not need the next generation tbh. I cannot fathom bringing a new slave into this world when this gene...",
            "Doing research and getting survey responses is actually hell",
            "Everyone in this dorm cannot stand you and wishes you the worst. You are the rudest most inconsiderate person imaginable...",
            "I have already been suicidal for more than four years and it has become way too overwhelming for somebody with severe AD...",
            "I am feeling lost as a senior in college and compsci major",
            "feel shame for my grade, and I am doing the worst in my class.",
            'Feeling Disconnected From the "Chasing Girls" Culture',
            "I am just frustrated i maybe talking nonsense or I may not make sense but i absolutely hate FSD subject out of all. And ...",
            "So, I need help, my group assignment partners are not doing shit. They are not answering they are not doing anything the...",
            "Anyone else tired of discussions, its Thursday, when they are usually due. And I am honestly procrastinating so bad.",
        ],
    })
    st.dataframe(
        top_posts.style.background_gradient(subset=["NAS"], cmap="Reds"),
        use_container_width=True, height=320,
    )

    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.5rem;'>
    """ +
    insight("📌 <b>Rentrée EN (0.481)</b> est la saison au NAS le plus élevé — début d'année académique perçu comme anxiogène.", "danger") +
    insight("🇫🇷 <b>FR systématiquement plus bas</b> — différences culturelles d'expression ou corpus FR moins orienté affect négatif.", "warning") +
    insight("📊 <b>r/CollegeRant NAS = 0.645</b> — communauté de venting, affect négatif nettement dominant.", "danger") +
    insight("📐 Formule : <code>NAS = NA / (NA + PA + ε)</code> · NA = fear + sadness + anger + disgust · PA = joy + trust + anticipation · Fondé sur PANAS [Watson et al., 1988].") +
    "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 5 — PATTERNS TEMPORELS
# ─────────────────────────────────────────────
elif page == "📈  Patterns temporels":
    st.markdown("<div class='section-title'>Analyse temporelle</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Patterns temporels</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Évolution mensuelle des émotions · Patterns hebdomadaires · Saisonnalité</div>", unsafe_allow_html=True)

    # Évolution mensuelle émotions
    st.markdown("**Évolution mensuelle des émotions (score moyen)**")
    monthly_emotions = {
        "Jan":  {"anger":0.097,"anticipation":0.207,"disgust":0.010,"fear":0.040,"joy":0.075,"sadness":0.108,"surprise":0.130,"trust":0.135},
        "Fév":  {"anger":0.101,"anticipation":0.204,"disgust":0.010,"fear":0.028,"joy":0.053,"sadness":0.091,"surprise":0.143,"trust":0.150},
        "Mar":  {"anger":0.088,"anticipation":0.228,"disgust":0.009,"fear":0.025,"joy":0.056,"sadness":0.076,"surprise":0.148,"trust":0.142},
        "Avr":  {"anger":0.104,"anticipation":0.188,"disgust":0.010,"fear":0.039,"joy":0.059,"sadness":0.130,"surprise":0.119,"trust":0.143},
        "Mai":  {"anger":0.088,"anticipation":0.233,"disgust":0.009,"fear":0.024,"joy":0.049,"sadness":0.088,"surprise":0.149,"trust":0.132},
        "Août": {"anger":0.080,"anticipation":0.196,"disgust":0.009,"fear":0.036,"joy":0.043,"sadness":0.110,"surprise":0.111,"trust":0.076},
        "Sep":  {"anger":0.101,"anticipation":0.184,"disgust":0.010,"fear":0.067,"joy":0.048,"sadness":0.151,"surprise":0.183,"trust":0.096},
        "Oct":  {"anger":0.083,"anticipation":0.153,"disgust":0.010,"fear":0.056,"joy":0.063,"sadness":0.139,"surprise":0.154,"trust":0.083},
        "Nov":  {"anger":0.080,"anticipation":0.149,"disgust":0.010,"fear":0.053,"joy":0.050,"sadness":0.142,"surprise":0.163,"trust":0.077},
        "Déc":  {"anger":0.111,"anticipation":0.190,"disgust":0.010,"fear":0.049,"joy":0.065,"sadness":0.146,"surprise":0.122,"trust":0.110},
    }
    df_monthly = pd.DataFrame(monthly_emotions).T.reset_index()
    df_monthly.columns = ["mois"] + list(df_monthly.columns[1:])
    df_melt = df_monthly.melt(id_vars="mois", var_name="emotion", value_name="score")
    order = ["Août","Sep","Oct","Nov","Déc","Jan","Fév","Mar","Avr","Mai"]
    df_melt["mois"] = pd.Categorical(df_melt["mois"], categories=order, ordered=True)
    df_melt = df_melt.sort_values("mois")

    fig = px.line(
        df_melt, x="mois", y="score", color="emotion",
        color_discrete_map=EMOTION_COLORS,
        markers=True,
        labels={"score":"Score moyen","mois":"","emotion":"Émotion"},
    )
    fig.update_traces(line_width=2)
    st.plotly_chart(styled_plotly(fig, 380), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Score sentiment par jour de la semaine**")
        jours = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        sent_en = [-0.134,-0.118,-0.128,-0.120,-0.111,-0.120,-0.098]
        sent_fr = [-0.119,-0.153,-0.055,-0.030,-0.060,-0.130,-0.093]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=jours, y=sent_en, name="EN", mode="lines+markers",
                                  line_color="#6C63FF", line_width=2))
        fig2.add_trace(go.Scatter(x=jours, y=sent_fr, name="FR", mode="lines+markers",
                                  line_color="#2EC4B6", line_width=2))
        fig2.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
        fig2.update_layout(xaxis_title="", yaxis_title="Score moyen")
        st.plotly_chart(styled_plotly(fig2, 300), use_container_width=True)

    with col2:
        st.markdown("**Volume de posts par jour**")
        vol_jours = [1401,1423,1412,1481,1263,1060,1162]
        fig3 = px.bar(
            x=jours, y=vol_jours,
            color=jours,
            color_discrete_sequence=["#6C63FF","#48CAE4","#F4A261","#E63946","#2EC4B6","#FFD166","#9B2335"],
            text=vol_jours,
            labels={"x":"","y":"Nb posts"},
        )
        fig3.update_traces(textposition="outside", textfont_color="#CBD5E1", marker_line_width=0)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(styled_plotly(fig3, 300), use_container_width=True)

    # Distress mensuel
    st.markdown("**NAS moyen par mois**")
    dist_en = [0.094,0.102,0.083,0.094,0.098,0.078,0.074,0.065,0.075,0.073]
    dist_fr = [None,None,None,None,None,0.012,0.038,0.039,0.035,0.039]
    mois_label = ["Août","Sep","Oct","Nov","Déc","Jan","Fév","Mar","Avr","Mai"]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=mois_label[:len(dist_en)], y=dist_en, name="EN",
                              mode="lines+markers", line_color="#6C63FF", line_width=2,
                              fill="tozeroy", fillcolor="rgba(108,99,255,0.1)"))
    fr_vals = [v for v in dist_fr if v is not None]
    fig4.add_trace(go.Scatter(x=mois_label[:len(fr_vals)], y=fr_vals, name="FR",
                              mode="lines+markers", line_color="#2EC4B6", line_width=2,
                              fill="tozeroy", fillcolor="rgba(46,196,182,0.1)"))
    fig4.update_layout(xaxis_title="", yaxis_title="NAS moyen")
    st.plotly_chart(styled_plotly(fig4, 280), use_container_width=True)

    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.6rem; margin-top:0.5rem;'>
    """ +
    insight("📅 <b>Anticipation domine sur toute l'année</b>, avec un pic en mai (examens).") +
    insight("📆 <b>Jeudi = pic de volume</b> (1 481 posts) ; samedi = creux (1 060).", "warning") +
    insight("🌡️ <b>NAS EN culminant en septembre–décembre</b>, corrélé aux périodes d'évaluation.", "danger") +
    "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 6 — GRAPHES & TOPICS
# ─────────────────────────────────────────────
elif page == "🕸️  Graphes & Topics":
    st.markdown("<div class='section-title'>Visualisation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Graphes & Topics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Co-occurrences NetworkX/Pyvis · BERTopic multilingue · 14 topics · 9 233 posts valides</div>", unsafe_allow_html=True)

    # Stats topics
    cols = st.columns(4)
    for col, (v,l,s) in zip(cols, [
        ("14","Topics identifiés","T0–T13, outliers réassignés"),
        ("3 861","Posts topicisés","41.8% du corpus valide"),
        ("paraphrase-multilingual","Modèle embeddings","MiniLM-L12-v2"),
        ("UMAP + HDBSCAN","Algorithme clustering","leaf method"),
    ]):
        col.markdown(metric_card(v,l,s), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Graphes HTML
    graphs_dir = Path("reports/graphs/html")
    available_graphs = {
        "Co-occurrences EN":    "cooc_en.html",
        "Co-occurrences FR":    "cooc_fr.html",
        "Co-occurrences Global":"cooc_all.html",
        "Topics ↔ Mots-clés":  "topics_words.html",
    }

    graph_choice = st.selectbox(
        "Sélectionner un graphe interactif",
        list(available_graphs.keys()),
        label_visibility="visible"
    )

    graph_path = graphs_dir / available_graphs[graph_choice]

    if graph_path.exists():
        with open(graph_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.markdown(f"""
        <div style='background:rgba(108,99,255,0.08); border:1px dashed #6C63FF;
             border-radius:12px; padding:2.5rem; text-align:center; color:#94A3B8;'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>🕸️</div>
            <div style='font-family:Space Mono,monospace; font-size:0.85rem; color:#6C63FF;'>
                {available_graphs[graph_choice]}
            </div>
            <div style='margin-top:0.5rem; font-size:0.82rem;'>
                Fichier non trouvé dans <code>reports/graphs/html/</code><br>
                Placez les fichiers HTML générés par <code>graph.py</code> pour activer la visualisation interactive.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Distribution topics — vraies données Annexe III
    st.markdown("**Distribution des topics (BERTopic — 3 861 posts assignés)**")
    topics_data = {
        "T0 — General Academic Life & Wellbeing": {"count": 1402, "nas": 0.484, "pct_high": 57.8, "lang": "EN"},
        "T1 — Internship & Career Recruitment":   {"count": 472,  "nas": 0.200, "pct_high": 16.3, "lang": "EN"},
        "T2 — Research Surveys":                  {"count": 440,  "nas": 0.234, "pct_high": 27.5, "lang": "EN/FR"},
        "T3 — French Student Life (General)":     {"count": 388,  "nas": 0.250, "pct_high": 28.1, "lang": "FR"},
        "T4 — French Academic Orientation":       {"count": 295,  "nas": 0.299, "pct_high": 38.6, "lang": "FR"},
        "T5 — International Students in France":  {"count": 220,  "nas": 0.209, "pct_high": 22.3, "lang": "FR"},
        "T6 — Examinations & Academic Stress":    {"count": 146,  "nas": 0.425, "pct_high": 50.0, "lang": "EN"},
        "T7 — Sleep, Fatigue & Mental Load":      {"count": 122,  "nas": 0.340, "pct_high": 36.1, "lang": "EN"},
        "T8 — Student Housing (CROUS)":           {"count": 88,   "nas": 0.239, "pct_high": 25.0, "lang": "FR"},
        "T10 — Sciences & Mathematics (FR)":      {"count": 79,   "nas": 0.294, "pct_high": 31.6, "lang": "FR"},
        "T9 — Business School & Competitions":    {"count": 75,   "nas": 0.133, "pct_high": 6.7,  "lang": "EN"},
        "T11 — Student Discounts & Finances":     {"count": 52,   "nas": 0.099, "pct_high": 1.9,  "lang": "EN"},
        "T12 — Academic Resources & Textbooks":   {"count": 46,   "nas": 0.207, "pct_high": 19.6, "lang": "EN"},
        "T13 — Summer Internship Housing":        {"count": 36,   "nas": 0.092, "pct_high": 0.0,  "lang": "EN"},
    }
    df_topics_viz = pd.DataFrame([
        {"topic": k, "count": v["count"], "nas": v["nas"], "pct_high": v["pct_high"], "lang": v["lang"]}
        for k, v in topics_data.items()
    ]).sort_values("count", ascending=False)

    fig = px.bar(
        df_topics_viz, x="topic", y="count",
        color="nas",
        color_continuous_scale=["#1A535C","#6C63FF","#F4A261","#E63946"],
        text="count",
        labels={"count":"Nb posts","topic":"","nas":"NAS moyen"},
        hover_data={"nas":":.3f","pct_high":":.1f","lang":True},
    )
    fig.update_traces(textposition="outside", textfont_color="#CBD5E1", marker_line_width=0)
    fig.update_layout(showlegend=False, xaxis_tickangle=-35, coloraxis_showscale=True,
                      coloraxis_colorbar=dict(title="NAS moyen", thickness=12, len=0.6))
    st.plotly_chart(styled_plotly(fig, 400), use_container_width=True)

    # NAS moyen par topic
    st.markdown("**NAS moyen par topic (% high NAS)**")
    df_nas_topic = df_topics_viz.sort_values("nas")
    fig_nas = px.bar(
        df_nas_topic, x="nas", y="topic", orientation="h",
        color="nas",
        color_continuous_scale=["#1A535C","#6C63FF","#F4A261","#E63946"],
        text=df_nas_topic["nas"].apply(lambda x: f"{x:.3f}"),
        labels={"nas":"NAS moyen","topic":""},
    )
    fig_nas.add_vline(x=0.4, line_dash="dash", line_color="#F4A261",
                      annotation_text="Seuil high (0.4)", annotation_font_color="#F4A261")
    fig_nas.update_traces(textposition="outside", textfont_color="#CBD5E1", marker_line_width=0)
    fig_nas.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(styled_plotly(fig_nas, 420), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='insight-box danger'>
            🔥 <b>T0 (General Academic Life & Wellbeing)</b> : NAS = 0.484, 57.8% high — le topic le plus chargé émotionnellement.
        </div>
        <div class='insight-box warning'>
            📝 <b>T6 (Examinations & Academic Stress)</b> : NAS = 0.425, 50% high — stress d'examen bien documenté.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='insight-box success'>
            💶 <b>T11, T13</b> (Finances, Housing) : NAS ≈ 0.09 — topics fonctionnels, quasi sans affect négatif.
        </div>
        <div class='insight-box'>
            🔗 <b>Graphes de co-occurrences :</b> nœuds = mots lemmatisés, taille = fréquence, couleur = émotion dominante.
        </div>
        """, unsafe_allow_html=True)