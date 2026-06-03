import pandas as pd

df = pd.read_csv("/Users/amalamyay/Desktop/thesis-mental-health-NLP/data/processed/reddit_emotions.csv")

NAS_NEGATIVE = ["fear", "sadness", "anger", "disgust"]
NAS_POSITIVE = ["joy", "trust", "anticipation"]

eps = 1e-8
na = sum(df[f"emotion_{e}"].fillna(0) for e in NAS_NEGATIVE)
pa = sum(df[f"emotion_{e}"].fillna(0) for e in NAS_POSITIVE)
df["nas_score"] = (na / (na + pa + eps)).round(4)

df["nas_level"] = pd.qcut(
    df["nas_score"],
    q=[0, 0.33, 0.66, 1.0],
    labels=["low", "moderate", "high"],
    duplicates="drop"
)

df.to_csv("/Users/amalamyay/Desktop/thesis-mental-health-NLP/data/processed/reddit_emotions.csv", index=False, encoding="utf-8-sig")
print("Done")