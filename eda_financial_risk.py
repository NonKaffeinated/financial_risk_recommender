import os
import json
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

DATASET_PATH = os.environ.get("DATASET_PATH", "financial-news-dataset/Datasets/extracted")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER_TO_NAME = {
    "AAPL": "apple",
    "TSLA": "tesla",
    "MSFT": "microsoft",
    "NVDA": "nvidia",
    "GOOGL": "google",
    "AMZN": "amazon",
    "META": "meta facebook",
    "JPM": "jpmorgan",
    "NFLX": "netflix",
}

FINANCIAL_FEATURES = [
    "debtToEquity",
    "profitMargins",
    "revenueGrowth",
    "currentRatio",
    "returnOnEquity",
    "operatingMargins",
]

BASELINE_TICKERS = ["AAPL", "MSFT", "JNJ", "JPM", "PG", "V", "UNH", "HD", "MA", "DIS"]


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
        title = article.get("title", "") or ""
        text = article.get("text", "") or article.get("title", "") or ""
        label = infer_label(filepath)
        rows.append(
            {
                "filepath": filepath,
                "label": label,
                "label_name": {1: "Positive", 0: "Negative", 2: "Neutral"}[label],
                "title": title,
                "text": text,
                "title_len": len(title),
                "text_len": len(text),
                "has_title": bool(title),
                "has_text": bool(text),
                "fields_present": len(article.keys()),
                "keys": sorted(article.keys()),
            }
        )
    return pd.DataFrame(rows)


def schema_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["field", "presence_ratio"])
    key_counter = Counter()
    total = len(df)
    for keys in df["keys"]:
        key_counter.update(keys)
    rows = [{"field": k, "presence_ratio": round(v / total, 4)} for k, v in key_counter.items()]
    return pd.DataFrame(rows).sort_values(["presence_ratio", "field"], ascending=[False, True])


def ticker_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "keyword", "matches"])
    corpus = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()
    rows = []
    for ticker, keyword in TICKER_TO_NAME.items():
        for token in keyword.split():
            rows.append(
                {
                    "ticker": ticker,
                    "keyword": token,
                    "matches": int(corpus.str.contains(token, regex=False).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_financial_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, **{feature: np.nan for feature in FINANCIAL_FEATURES}}
            for ticker in BASELINE_TICKERS
        ]
    )


def main():
    news_df = load_news_articles(DATASET_PATH)
    schema_df = schema_profile(news_df)
    coverage_df = ticker_coverage(news_df)
    financial_df = build_financial_template()

    if not news_df.empty:
        news_df[["filepath", "label_name", "title_len", "text_len", "fields_present"]].head(200).to_csv(
            OUTPUT_DIR / "news_sample_rows.csv", index=False
        )
        news_df["label_name"].value_counts(dropna=False).rename_axis("label").reset_index(name="count").to_csv(
            OUTPUT_DIR / "news_label_counts.csv", index=False
        )
        pd.DataFrame(
            {
                "metric": [
                    "records",
                    "has_title_rate",
                    "has_text_rate",
                    "avg_title_len",
                    "avg_text_len",
                    "median_text_len",
                ],
                "value": [
                    len(news_df),
                    round(float(news_df["has_title"].mean()), 4),
                    round(float(news_df["has_text"].mean()), 4),
                    round(float(news_df["title_len"].mean()), 2),
                    round(float(news_df["text_len"].mean()), 2),
                    round(float(news_df["text_len"].median()), 2),
                ],
            }
        ).to_csv(OUTPUT_DIR / "news_quality_metrics.csv", index=False)

    schema_df.to_csv(OUTPUT_DIR / "news_schema_profile.csv", index=False)
    coverage_df.to_csv(OUTPUT_DIR / "ticker_keyword_coverage.csv", index=False)
    financial_df.to_csv(OUTPUT_DIR / "financial_feature_template.csv", index=False)

    report = []
    report.append("# FinSage Dataset EDA\n")
    report.append(f"Dataset path: `{DATASET_PATH}`\n")
    if news_df.empty:
        report.append(
            "No JSON news articles were found locally. This script still produced schema and financial templates so the analysis can be rerun once the Webhose dataset is extracted.\n"
        )
    else:
        report.append(f"Loaded {len(news_df)} news records from the Webhose financial-news-dataset.\n")
        report.append("## Label distribution\n")
        report.append(
            news_df["label_name"].value_counts(dropna=False).rename_axis("label").reset_index(name="count").to_markdown(index=False)
            + "\n"
        )
        report.append("## Schema profile\n")
        report.append(schema_df.head(20).to_markdown(index=False) + "\n")
        report.append("## Data quality\n")
        quality = pd.DataFrame(
            {
                "metric": [
                    "has_title_rate",
                    "has_text_rate",
                    "avg_title_len",
                    "avg_text_len",
                    "median_text_len",
                ],
                "value": [
                    round(float(news_df["has_title"].mean()), 4),
                    round(float(news_df["has_text"].mean()), 4),
                    round(float(news_df["title_len"].mean()), 2),
                    round(float(news_df["text_len"].mean()), 2),
                    round(float(news_df["text_len"].median()), 2),
                ],
            }
        )
        report.append(quality.to_markdown(index=False) + "\n")
        report.append("## Ticker coverage\n")
        report.append(coverage_df.groupby("ticker", as_index=False)["matches"].sum().to_markdown(index=False) + "\n")

    report.append("## Financial feature template\n")
    report.append(
        "The financial model uses live Yahoo Finance fields instead of a committed training file, so this EDA exports the expected six-feature template for baseline tickers.\n"
    )
    report.append(financial_df.to_markdown(index=False) + "\n")

    (OUTPUT_DIR / "eda_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()