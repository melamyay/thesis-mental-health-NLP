# thesis-mental-health-nlp

NLP pipeline for analyzing student mental health from online forum data.

> Master's thesis — ILIS UFR3S — PROMO 2026

**Live Dashboard → [mental-health-nlp.streamlit.app](https://mental-health-nlp.streamlit.app/)**
---

## Objective

Design and implement a reproducible pipeline for processing and analyzing textual data from online forums, enabling the extraction of mental health indicators and their representation as a knowledge graph, in order to explore their temporal dynamics across an academic curriculum.
 
**Research question:** How can textual data from online forums be exploited, using NLP methods and knowledge graph modelling, to characterize the evolution of student mental health over the course of an academic curriculum?
 

---

## ️ Project structure

```
thesis-mental-health-NLP/
├── data/
│   ├── raw/                        # Raw collected data
│   │   ├── forums_reddit/          # First scraping batch
│   │   └── forums_reddit2/         # Second scraping batch (auto-detected)
│   │   └── forums_reddit*/         # Additional scraping batch 
│   ├── processed/                  # Cleaned and enriched data
│   │   ├── reddit_clean.csv        # After preprocessing
│   │   ├── reddit_sentiment.csv    # + sentiment scores
│   │   ├── reddit_emotions.csv     # + emotion scores + distress index
│   │   └── reddit_topics.csv       # + topic labels (BERTopic)
│   ├── aggregated/                 # Temporally aggregated indicators
│   │   ├── agg_monthly.csv
│   │   ├── agg_seasonal.csv
│   │   ├── agg_dayofweek.csv
│   │   └── agg_global.csv
│   └── lexicons/                   # NRC Emotion Lexicon (auto-downloaded)
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py           # Cleaning, tokenization, lemmatization
│   │   └── validate.py             # Dataset quality report + figures
│   ├── features/
│   │   ├── sentiment.py            # CamemBERT (FR) + RoBERTa (EN)
│   │   ├── emotions.py             # NRC lexicon + go-emotions BERT + distress score
│   │   ├── topics.py               # BERTopic multilingual
│   │   ├── aggregation.py          # Temporal aggregation (month/season/day)
│   │   └── visualize.py            # Analytical figures (matplotlib)
│   └── visualization/
│       └── graph.py                # Co-occurrence graphs (NetworkX + pyvis)
├── reports/
│   ├── figures/                    # Generated PNG figures
│   └── graphs/
│       ├── html/                   # Interactive pyvis graphs
│       └── png/                    # Static graph exports
├── notebooks/                      # Exploration notebooks
├── config/                         # Configuration files
├── tests/                          # Unit tests
├── dashboard.py                    # Streamlit interactive dashboard
├── main.py                         # Full pipeline orchestrator
├── requirements.txt                # Dashboard dependencies (Streamlit Cloud)
├── requirements_full.txt           # Full pipeline dependencies
├── .gitignore
└── README.md
```

---

## ️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/melamyay>/thesis-mental-health-NLP.git
cd thesis-mental-health-NLP
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
pip install -r requirements_full.txt
```

### 4. Download spaCy models
 
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```
 
---
 
## Usage
 
### Run the full pipeline (one command)
 
```bash
python main.py
```

### Options
 
```bash
# Preprocessing only (no NLP)
python main.py --skip_nlp
 
# Skip BERTopic (recommended on Mac — llvmlite issues)
python main.py --skip_topics
 
# Custom input/output directories
python main.py --input_dir data/raw --output_dir data/processed
```

## Run individual modules
 
```bash
# 1. Preprocessing
python src/preprocessing/preprocess.py
 
# 2. Validation report + figures
python src/preprocessing/validate.py
 
# 3. Sentiment analysis
python src/features/sentiment.py
 
# 4. Emotion extraction + distress index
python src/features/emotions.py
 
# 5. Topic modelling (BERTopic)
python src/features/topics.py
 
# 6. Temporal aggregation
python src/features/aggregation.py
 
# 7. Analytical figures
python src/features/visualize.py
 
# 8. Co-occurrence graphs
python src/visualization/graph.py
```

### Run the dashboard locally
```bash
streamlit run dashboard.py
```

---

##  Pipeline

```
Raw XLSX files (forums_reddit*, auto-detected)
        ↓
Preprocessing
  · URL removal, emoji removal, contraction expansion (EN/FR)
  · Automatic language detection (langdetect)
  · Tokenization + lemmatization (spaCy EN/FR)
  · Deduplication by id_post across all scraping batches
        ↓
Sentiment Analysis
  · CamemBERT for French posts
  · RoBERTa for English posts
  · Output: positive / neutral / negative + continuous score [-1, 1]
        ↓
Emotion Extraction
  · NRC Emotion Lexicon (30%) + go-emotions BERT (70%)
  · 8 emotions (Plutchik's wheel): anger, anticipation, disgust,
    fear, joy, sadness, surprise, trust
  · Distress index = (fear + sadness + 0.5×anger) / 2.5 → [0, 1]
        ↓
Topic Modelling
  · BERTopic with paraphrase-multilingual-MiniLM-L12-v2
  · Multilingual FR+EN, 12 topics
        ↓
Temporal Aggregation
  · By month, academic season, day of week
  · By language (EN/FR) and subreddit
        ↓
Visualization
  · Analytical figures (sentiment, emotions, distress over time)
  · Interactive dashboard -> https://mental-health-nlp.streamlit.app/
```

---
 
## Key Results
 
| Indicator | Value |
|---|---|
| Corpus size | 10,412 posts |
| Negative sentiment | 37.5% |
| Mean score_cont EN | -0.25 |
| Mean score_cont FR | -0.12 |
| Dominant emotion | Anticipation (32.8%) |
| Mean distress score | 0.068 |
| Moderate distress posts | 743 (8.1%) |
| Topics identified | 12 |
| Peak distress period | Rentree EN (0.107) |
 
---

## Data Sources
 
| Subreddit | Language | Posts |
|---|---|---|
| r/Students | English | 1,998 |
| r/etudiants | French | 1,991 |
| r/csMajors | English | 1,990 |
| r/CollegeRant | English | 1,886 |
| r/Student | English | 1,780 |
| r/college | English | 767 |
| **Total** | EN: 8,275 / FR: 1,939 | **10,412** |
 
Collection period: August 2025 → March 2026
 
---

##  Tech stack

| Layer | Tool |
|-------|------|
| Data collection | PRAW (Reddit API) |
| Preprocessing | spaCy, langdetect, emoji |
| Sentiment | HuggingFace Transformers (CamemBERT, RoBERTa) |
| Emotions | NRC Lexicon, go-emotions BERT |
| Topic modelling | BERTopic, sentence-transformers |
| Aggregation | pandas |
| Visualization | matplotlib, NetworkX, pyvis |
| Dashboard | Streamlit, Plotly |
 
---

## 👥 Contributors

- **AMYAY Amal** — [@melamyay](https://github.com/melamyay)
- **COKELAER Alexis** — [@alexiscokelaer](https://github.com/alexiscokelaer)

**Supervisor:** Antoine LAMER

---

##  License

This project was developed as part of an academic Master's thesis. Any reuse must credit the original authors.
