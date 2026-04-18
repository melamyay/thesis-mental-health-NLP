# thesis-mental-health-nlp

NLP pipeline for analyzing student mental health from online forum data.

> Master's thesis — ILIS UFR3S — PROMO 2026

---

## Objective

Design and implement a reproducible pipeline for processing and analyzing textual data from online forums, enabling the extraction of mental health indicators and their representation as a knowledge graph, in order to explore their temporal dynamics across an academic curriculum.

---

## ️ Project structure

```
thesis-mental-health-nlp/
├── data/
│   ├── raw/              # Raw collected data (forums)
│   └── processed/        # Cleaned and transformed data
├── src/
│   ├── scraping/         # Data collection (Reddit API, etc.)
│   ├── preprocessing/    # Cleaning, tokenization, normalization
│   ├── nlp/              # Sentiment analysis, topic modeling, etc.
│   └── knowledge_graph/  # Neo4j modeling and queries
├── notebooks/            # Exploration and visualizations
├── outputs/              # Results, figures, exports
├── tests/                # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/melamyay>/thesis_mental_health_NLP.git
cd thesis_mental_health_NLP
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up credentials

Copy the example file and fill in your API keys:

```bash
cp config/secrets.example.yml config/secrets.yml
# Edit secrets.yml with your Reddit credentials, etc.
```

---

##  Pipeline

```
Data collection (Reddit/forums)
        ↓
Preprocessing (cleaning, tokenization)
        ↓
NLP analysis (sentiment, topics, entities)
        ↓
Knowledge graph (Neo4j)
        ↓
Temporal analysis & visualization
```

---

##  Tech stack

| Layer | Tool |
|-------|------|
| Data collection | PRAW (Reddit API) |
| NLP | spaCy, CamemBERT, HuggingFace Transformers |
| Topic modeling | Gensim (LDA) |
| Knowledge graph | Neo4j |
| Visualization | Plotly, NetworkX |

---

## 👥 Contributors

- AMYAY AMAL — [@melamyay]
- COKELAER ALEXIS — [@alexiscokelaer]

---

##  License

This project was developed as part of an academic thesis. Any reuse must credit the original authors.
