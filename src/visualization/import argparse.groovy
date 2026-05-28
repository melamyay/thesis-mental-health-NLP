import argparse
import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

HTML_DIR = Path("reports/graphs/html")
PNG_DIR  = Path("reports/graphs/png")

# Couleurs par émotion dominante
EMOTION_COLORS = {
    "anger":        "#E24B4A",
    "anticipation": "#BA7517",
    "disgust":      "#712B13",
    "fear":         "#A32D2D",
    "joy":          "#1D9E75",
    "sadness":      "#534AB7",
    "surprise":     "#EF9F27",
    "trust":        "#085041",
    None:           "#888780",
}

# Mots à exclure des graphes (trop génériques)
STOPWORDS_GRAPH = {
    "like", "get", "got", "know", "want", "thing", "think", "go", "going",
    "make", "way", "time", "good", "really", "feel", "need", "use", "say",
    "said", "come", "look", "take", "give", "see", "try", "tell", "ask",
    "would", "could", "also", "even", "much", "many", "still", "back",
    "actually", "just", "people", "one", "lot", "bit", "sure", "right",
    "something", "someone", "anyone", "everyone", "hi", "hey", "thank",
    "thanks", "please", "lol", "idk", "imo", "tbh", "ngl",
    "vraiment", "faire", "savoir", "être", "avoir", "aller", "bonjour",
    "merci", "comme", "mais", "pour", "avec", "dans", "sur", "plus",
    "bien", "très", "tout", "aussi", "quand", "alors", "après", "avant",
}

TOP_N_WORDS    = 80    # nb de mots les plus fréquents à garder
MIN_COOC       = 5     # co-occurrence minimale pour créer une arête
MIN_WORD_FREQ  = 10    # fréquence minimale d'un mot pour apparaître
MAX_NODES      = 60    # max noeuds dans le graphe final
MAX_EDGES      = 200   # max arêtes dans le graphe final


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_dirs():
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def get_emotion_color(word_emotion: str) -> str:
    return EMOTION_COLORS.get(word_emotion, "#888780")


def word_to_emotion(word: str, df: pd.DataFrame) -> str:
    """Trouver l'émotion dominante associée à un mot via les posts qui le contiennent."""
    if "emotion_dominant" not in df.columns or "texte_lemmatise" not in df.columns:
        return None
    mask = df["texte_lemmatise"].fillna("").str.contains(r"\b" + word + r"\b", regex=True)
    subset = df[mask]["emotion_dominant"].dropna()
    if len(subset) == 0:
        return None
    return subset.value_counts().index[0]


# ── 1. Graphe de co-occurrences ───────────────────────────────────────────────

def build_cooc_graph(df: pd.DataFrame, lang: str = "all", top_n: int = TOP_N_WORDS) -> nx.Graph:
    """
    Construit un graphe de co-occurrences de mots.
    Noeuds = mots lemmatisés fréquents
    Arêtes = co-occurrence dans le même post (poids = nb de co-occurrences)
    Couleur noeud = émotion dominante des posts contenant ce mot
    Taille noeud = fréquence du mot
    """
    log.info(f"Building co-occurrence graph (lang={lang})...")

    # Filtrer par langue
    if lang != "all":
        sub = df[df["langue"] == lang].copy()
    else:
        sub = df.copy()

    # Tokeniser
    texts = sub["texte_lemmatise"].fillna("").astype(str).tolist()
    tokenized = []
    for text in texts:
        tokens = [
            t for t in text.lower().split()
            if len(t) > 2 and t not in STOPWORDS_GRAPH and t.isalpha()
        ]
        tokenized.append(tokens)

    # Fréquences des mots
    word_freq = Counter(w for tokens in tokenized for w in tokens)
    top_words = {w for w, c in word_freq.most_common(top_n) if c >= MIN_WORD_FREQ}
    log.info(f"  Top words: {len(top_words)}")

    # Compter les co-occurrences
    cooc = Counter()
    for tokens in tokenized:
        filtered = [t for t in tokens if t in top_words]
        filtered = list(set(filtered))  # dédupliquer dans le même post
        for w1, w2 in combinations(sorted(filtered), 2):
            cooc[(w1, w2)] += 1

    # Construire le graphe
    G = nx.Graph()

    # Ajouter les noeuds
    for word in top_words:
        freq = word_freq[word]
        emotion = word_to_emotion(word, sub)
        G.add_node(word, freq=freq, emotion=emotion, color=get_emotion_color(emotion))

    # Ajouter les arêtes (filtrer par poids minimum)
    edges_added = 0
    for (w1, w2), weight in sorted(cooc.items(), key=lambda x: -x[1]):
        if weight >= MIN_COOC and w1 in G and w2 in G:
            G.add_edge(w1, w2, weight=weight)
            edges_added += 1
        if edges_added >= MAX_EDGES:
            break

    # Garder seulement les noeuds connectés + les plus fréquents
    connected = set(n for n in G.nodes() if G.degree(n) > 0)
    isolated = set(G.nodes()) - connected
    G.remove_nodes_from(isolated)

    # Limiter le nb de noeuds
    if len(G.nodes()) > MAX_NODES:
        top_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["freq"], reverse=True)[:MAX_NODES]
        G = G.subgraph(top_nodes).copy()

    log.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, word_freq


def build_topic_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Graphe bipartite : Topics ↔ Mots clés
    Noeuds topics = grands noeuds colorés par taille (nb posts)
    Noeuds mots = petits noeuds
    Arêtes = appartenance mot → topic
    """
    log.info("Building topic graph...")

    if "topic_label" not in df.columns or "texte_lemmatise" not in df.columns:
        log.warning("topic_label or texte_lemmatise missing")
        return nx.Graph(), {}

    G = nx.Graph()
    topic_colors = [
        "#534AB7", "#E24B4A", "#1D9E75", "#BA7517", "#085041",
        "#A32D2D", "#EF9F27", "#378ADD", "#712B13", "#0F6E56",
        "#633806", "#185FA5"
    ]

    topics = df[df["topic_id"] != -1]["topic_label"].value_counts()

    for i, (topic, count) in enumerate(topics.items()):
        color = topic_colors[i % len(topic_colors)]
        G.add_node(topic, node_type="topic", size=count, color=color,
                   freq=count, is_topic=True)

    # Extraire les top mots par topic
    for topic_label in topics.index:
        topic_posts = df[df["topic_label"] == topic_label]["texte_lemmatise"].fillna("")
        word_counts = Counter()
        for text in topic_posts:
            tokens = [
                t for t in str(text).lower().split()
                if len(t) > 3 and t not in STOPWORDS_GRAPH and t.isalpha()
            ]
            word_counts.update(tokens)

        top_words = word_counts.most_common(8)
        topic_color = G.nodes[topic_label]["color"]

        for word, freq in top_words:
            node_id = f"{word}_{topic_label[:10]}"
            G.add_node(node_id, node_type="word", label=word, size=freq,
                       color=topic_color + "99", freq=freq, is_topic=False)
            G.add_edge(topic_label, node_id, weight=freq)

    log.info(f"  Topic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── 2. Export pyvis HTML ──────────────────────────────────────────────────────

def export_pyvis(G: nx.Graph, filename: str, title: str = "", word_freq: dict = None):
    """Exporte un graphe NetworkX en HTML interactif via pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        log.error("pyvis not installed. Run: pip install pyvis")
        return

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node, attrs in G.nodes(data=True):
        freq    = attrs.get("freq", 1)
        color   = attrs.get("color", "#888780")
        is_topic = attrs.get("is_topic", False)
        label   = attrs.get("label", str(node))

        # Taille proportionnelle à la fréquence
        if is_topic:
            size = max(30, min(70, freq / 30))
        else:
            size = max(10, min(35, freq / 5 if word_freq else freq))

        net.add_node(
            str(node),
            label=label if not is_topic else str(node)[:25],
            size=size,
            color=color,
            title=f"{node}\nFréquence: {freq}\nÉmotion: {attrs.get('emotion', 'N/A')}",
            font={"size": 12 if not is_topic else 16, "color": "white"},
            borderWidth=2 if is_topic else 1,
            borderWidthSelected=4,
        )

    for u, v, attrs in G.edges(data=True):
        weight = attrs.get("weight", 1)
        net.add_edge(
            str(u), str(v),
            width=max(0.5, min(5, weight / 10)),
            color={"color": "#ffffff33", "highlight": "#ffffff99"},
            title=f"Co-occurrence: {weight}",
        )

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true
      }
    }
    """)

    out_path = HTML_DIR / filename
    net.save_graph(str(out_path))
    log.info(f"HTML graph saved → {out_path}")


# ── 3. Export matplotlib PNG ──────────────────────────────────────────────────

def export_png(G: nx.Graph, filename: str, title: str = "", word_freq: dict = None):
    """Exporte un graphe NetworkX en PNG statique via matplotlib."""
    if G.number_of_nodes() == 0:
        log.warning(f"Empty graph, skipping PNG: {filename}")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Tailles et couleurs des noeuds
    node_sizes  = []
    node_colors = []
    for node in G.nodes():
        attrs = G.nodes[node]
        freq  = attrs.get("freq", 1)
        color = attrs.get("color", "#888780")
        is_topic = attrs.get("is_topic", False)
        size = max(200, min(3000, freq * (20 if is_topic else 5)))
        node_sizes.append(size)
        node_colors.append(color)

    # Épaisseur des arêtes
    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [max(0.3, min(3, w / max_w * 3)) for w in edge_weights]

    # Dessiner
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           alpha=0.3, edge_color="white")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=8, font_color="white", font_weight="bold")

    # Légende émotions
    seen_emotions = set(G.nodes[n].get("emotion") for n in G.nodes())
    patches = [
        mpatches.Patch(color=EMOTION_COLORS.get(e, "#888780"),
                       label=e if e else "unknown")
        for e in sorted(seen_emotions) if e
    ]
    if patches:
        ax.legend(handles=patches, loc="upper left", fontsize=8,
                  facecolor="#2a2a4e", labelcolor="white", framealpha=0.8)

    ax.set_title(title, color="white", fontsize=14, pad=15)
    ax.axis("off")
    fig.tight_layout()

    out_path = PNG_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    log.info(f"PNG graph saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graphes de co-occurrences et topics")
    parser.add_argument("--input",  type=Path,  default=Path("data/processed/reddit_topics.csv"))
    parser.add_argument("--type",   type=str,   default="all",
                        choices=["all", "cooc", "topics"],
                        help="Type de graphe à générer")
    parser.add_argument("--lang",   type=str,   default="all",
                        choices=["all", "en", "fr"],
                        help="Filtrer par langue")
    parser.add_argument("--sample", type=int,   default=None)
    args = parser.parse_args()

    setup_dirs()
    log.info("=== Starting graph generation ===")

    # Charger les données
    input_path = args.input
    if not input_path.exists():
        input_path = Path("data/processed/reddit_emotions.csv")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df):,} posts from {input_path}")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    # ── Graphes co-occurrences ─────────────────────────────────────────────
    if args.type in ["all", "cooc"]:

        langs = ["en", "fr"] if args.lang == "all" else [args.lang]

        for lang in langs:
            G, word_freq = build_cooc_graph(df, lang=lang)
            if G.number_of_nodes() == 0:
                log.warning(f"Empty graph for lang={lang}, skipping")
                continue

            export_pyvis(G, f"cooc_{lang}.html",
                         title=f"Co-occurrences — {lang.upper()}",
                         word_freq=word_freq)
            export_png(G, f"cooc_{lang}.png",
                       title=f"Graphe de co-occurrences ({lang.upper()})\nTaille = fréquence · Couleur = émotion dominante",
                       word_freq=word_freq)

        # Graphe global (EN + FR ensemble)
        if args.lang == "all":
            G_all, word_freq_all = build_cooc_graph(df, lang="all")
            export_pyvis(G_all, "cooc_all.html", title="Co-occurrences — All languages")
            export_png(G_all, "cooc_all.png",
                       title="Graphe de co-occurrences (EN + FR)\nTaille = fréquence · Couleur = émotion dominante")

    # ── Graphe topics ──────────────────────────────────────────────────────
    if args.type in ["all", "topics"]:
        G_topics = build_topic_graph(df)
        if G_topics.number_of_nodes() > 0:
            export_pyvis(G_topics, "topics_words.html", title="Topics & mots clés")
            export_png(G_topics, "topics_words.png",
                       title="Graphe Topics ↔ Mots clés\nTaille = nb posts · Couleur = topic")

    log.info(f"=== Done — graphs saved to {HTML_DIR} and {PNG_DIR} ===")


if __name__ == "__main__":
    main()