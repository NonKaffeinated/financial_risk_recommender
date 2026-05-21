import os
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

DATASET_PATH = os.environ.get("DATASET_PATH", "financial-news-dataset/Datasets/extracted")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def infer_label(filepath: str) -> int:
    low = filepath.lower()
    if "positive" in low:
        return 1
    if "negative" in low:
        return 0
    return 2


def load_news_articles(path: str) -> pd.DataFrame:
    rows = []
    for filepath in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                article = json.load(f)
        except Exception:
            continue
        text = article.get("text", "") or article.get("title", "") or ""
        rows.append({"filepath": filepath, "label": infer_label(filepath), "text": text[:1000]})
    return pd.DataFrame(rows)


def benchmark_sentiment_models(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty or news_df["label"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "model": "sentiment_models",
                    "status": "skipped",
                    "reason": "dataset unavailable or only one class present",
                }
            ]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        news_df["text"], news_df["label"], test_size=0.2, random_state=42, stratify=news_df["label"]
    )
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    models = {
        "tfidf_logreg_baseline": LogisticRegression(max_iter=1000),
        "tfidf_random_forest_benchmark": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, model in models.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        row = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
        }
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(Xte)
                row["roc_auc_ovr_weighted"] = round(
                    float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")), 4
                )
            except Exception:
                row["roc_auc_ovr_weighted"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluation_gaps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "module": "financial.py",
                "current_state": "Isolation Forest on six Yahoo Finance ratios and ten baseline tickers",
                "still_needed": "Offline labeled benchmark dataset, holdout design, and comparison against Logistic Regression, Random Forest, and XGBoost",
                "recommended_metrics": "ROC-AUC, PR-AUC, recall for risky firms, calibration, confusion matrix",
            },
            {
                "module": "risk.py",
                "current_state": "Rule-based weighted blend plus prototype content-based method",
                "still_needed": "Frozen benchmark table mapping company snapshots to trust or risk labels and a comparison of rule-based vs content-based scoring",
                "recommended_metrics": "Risk-band accuracy, MAE on trust score, Spearman ranking correlation",
            },
            {
                "module": "sentiment.py advanced path",
                "current_state": "FinBERT stub exists but is not benchmarked",
                "still_needed": "Side-by-side evaluation against TF-IDF baseline on the same splits and ticker-group split",
                "recommended_metrics": "Weighted F1, per-class recall, confusion matrix, inference time",
            },
        ]
    )


def main():
    news_df = load_news_articles(DATASET_PATH)
    benchmark_df = benchmark_sentiment_models(news_df)
    gaps_df = evaluation_gaps()

    benchmark_df.to_csv(OUTPUT_DIR / "sentiment_benchmark_results.csv", index=False)
    gaps_df.to_csv(OUTPUT_DIR / "model_evaluation_gaps.csv", index=False)

    report = []
    report.append("# FinSage Benchmark and Evaluation\n")
    report.append(f"Dataset path: `{DATASET_PATH}`\n")
    if news_df.empty:
        report.append("No local Webhose dataset was found, so executable sentiment benchmarking was skipped.\n")
    else:
        report.append(f"Loaded {len(news_df)} records for sentiment benchmarking.\n")
    report.append("## Sentiment benchmark\n")
    report.append(benchmark_df.to_markdown(index=False) + "\n")
    report.append("## Models still needing evaluation work\n")
    report.append(gaps_df.to_markdown(index=False) + "\n")
    report.append("## Notes\n")
    report.append(
        "The sentiment model can be benchmarked directly once the Webhose dataset is available locally. The financial and composite risk models still need a labeled benchmark set because the current implementation depends on live Yahoo Finance pulls and heuristic blending rather than a frozen supervised evaluation dataset.\n"
    )

    (OUTPUT_DIR / "benchmark_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()

    