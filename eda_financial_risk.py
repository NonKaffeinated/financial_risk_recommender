import os
import re
import json
import glob
import zipfile
from pathlib import Path
from collections import Counter


try:
  import yfinance as yf
  _YFINANCE_AVAILABLE = True
except ImportError:
  _YFINANCE_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/content/drive/MyDrive/financial-news-dataset-master/Datasets",
)
OUTPUT_DIR = Path("/content/drive/MyDrive/financial-news-dataset-master/output_eda")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER_TO_NAME = {
    "AAPL":  "apple",
    "TSLA":  "tesla",
    "MSFT":  "microsoft",
    "NVDA":  "nvidia",
    "GOOGL": "google",
    "AMZN":  "amazon",
    "META":  "meta facebook",
    "JPM":   "jpmorgan",
    "NFLX":  "netflix",
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


def _label_from_zip(zip_name: str) -> tuple:
    low = zip_name.lower()
    if "positive" in low:
        return 1, "Positive"
    if "negative" in low:
        return 0, "Negative"
    return -1, "Unknown"


def _release_date_from_zip(zip_name: str) -> str:
    m = re.search(r"_(\d{8})\d{6}\.zip$", zip_name)
    return m.group(1) if m else ""


def load_news_articles(path: str) -> pd.DataFrame:
    rows = []
    zip_files = sorted(glob.glob(os.path.join(path, "*.zip")))
    for zip_path in zip_files:
        basename = os.path.basename(zip_path)
        label, label_name = _label_from_zip(basename)
        if label == -1:
            continue
        release_date = _release_date_from_zip(basename)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if not member.endswith(".json"):
                        continue
                    try:
                        with zf.open(member) as f:
                            raw = json.load(f)
                    except Exception:
                        continue
                    articles = raw if isinstance(raw, list) else [raw]
                    for a in articles:
                        if not isinstance(a, dict):
                            continue
                        title = a.get("title", "") or ""
                        text  = a.get("text",  "") or title
                        rows.append({
                            "zip_file":       basename,
                            "release_date":   release_date,
                            "label":          label,
                            "label_name":     label_name,
                            "title":          title,
                            "text":           text,
                            "title_len":      len(title),
                            "text_len":       len(text),
                            "has_title":      bool(title),
                            "has_text":       bool(a.get("text", "")),
                            "fields_present": len(a.keys()),
                            "keys":           sorted(a.keys()),
                        })
        except Exception:
            continue
    return pd.DataFrame(rows)


def schema_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["field", "presence_ratio"])
    key_counter = Counter()
    for keys in df["keys"]:
        key_counter.update(keys)
    total = len(df)
    rows = [
        {"field": k, "presence_ratio": round(v / total, 4)}
        for k, v in key_counter.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["presence_ratio", "field"], ascending=[False, True]
    ).reset_index(drop=True)


def ticker_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "keyword", "Positive", "Negative", "total"])
    corpus = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()
    rows = []
    for ticker, keywords in TICKER_TO_NAME.items():
        for token in keywords.split():
            mask = corpus.str.contains(token, regex=False)
            pos  = int((mask & (df["label_name"] == "Positive")).sum())
            neg  = int((mask & (df["label_name"] == "Negative")).sum())
            rows.append({
                "ticker":   ticker,
                "keyword":  token,
                "Positive": pos,
                "Negative": neg,
                "total":    pos + neg,
            })
    return (
        pd.DataFrame(rows)
        .sort_values(["ticker", "total"], ascending=[True, False])
        .reset_index(drop=True)
    )


def text_length_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pcts = [10, 25, 50, 75, 90, 95, 99]
    rows = []
    for col in ["title_len", "text_len"]:
        row = {"field": col, "mean": round(float(df[col].mean()), 1)}
        for p in pcts:
            row[f"p{p}"] = round(float(np.percentile(df[col], p)), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def zip_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["zip_file", "label_name", "release_date"])
        .size()
        .reset_index(name="articles")
        .sort_values("release_date")
        .reset_index(drop=True)
    )


def monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.assign(month=df["release_date"].str[:6])
        .groupby(["month", "label_name"])
        .size()
        .reset_index(name="articles")
        .sort_values(["month", "label_name"])
        .reset_index(drop=True)
    )


def save_plots(df: pd.DataFrame, month_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    palette = {"Positive": "#2ecc71", "Negative": "#e74c3c"}

    # label distribution bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label_name"].value_counts()
    bar_colors = [palette.get(l, "#95a5a6") for l in counts.index]
    bars = ax.bar(counts.index, counts.values, color=bar_colors)
    ax.set_title("Label Distribution")
    ax.set_ylabel("Article count")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_distribution.png", dpi=120)
    plt.close(fig)

    # text length histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    for lname, color in palette.items():
        sub = df[df["label_name"] == lname]["text_len"].clip(upper=10_000)
        ax.hist(sub, bins=60, alpha=0.6, label=lname, color=color)
    ax.set_title("Article Text Length Distribution")
    ax.set_xlabel("Characters (clipped at 10,000)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "text_length_distribution.png", dpi=120)
    plt.close(fig)

    # monthly volume line chart
    if not month_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        for lname, color in palette.items():
            sub = (
                month_df[month_df["label_name"] == lname]
                .set_index("month")["articles"]
            )
            ax.plot(sub.index, sub.values, marker="o", label=lname, color=color, linewidth=2)
        ax.set_title("Monthly Article Volume by Sentiment")
        ax.set_xlabel("Month (YYYYMM)")
        ax.set_ylabel("Articles")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "monthly_volume.png", dpi=120)
        plt.close(fig)

def _fetch_financial_features(tickers: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        row = {"ticker": t}
        if _YFINANCE_AVAILABLE:
            try:
                info = yf.Ticker(t).info
                for feat in FINANCIAL_FEATURES:
                    row[feat] = info.get(feat)
            except Exception:
                for feat in FINANCIAL_FEATURES:
                    row[feat] = np.nan
        else:
            for feat in FINANCIAL_FEATURES:
                row[feat] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    print(f"[eda] Loading articles from {DATASET_PATH} ...")
    news_df  = load_news_articles(DATASET_PATH)
    schema_df = schema_profile(news_df)
    cover_df  = ticker_coverage(news_df)
    pct_df    = text_length_percentiles(news_df)
    inv_df    = zip_inventory(news_df)
    month_df  = monthly_volume(news_df)
    fin_df = _fetch_financial_features(BASELINE_TICKERS)


    schema_df.to_csv(OUTPUT_DIR / "news_schema_profile.csv",        index=False)
    cover_df.to_csv(OUTPUT_DIR  / "ticker_keyword_coverage.csv",    index=False)
    fin_df.to_csv(OUTPUT_DIR    / "financial_feature_template.csv", index=False)

    if not news_df.empty:
        news_df[[
            "zip_file", "release_date", "label_name",
            "title_len", "text_len", "fields_present",
        ]].head(200).to_csv(OUTPUT_DIR / "news_sample_rows.csv", index=False)

        (
            news_df["label_name"]
            .value_counts(dropna=False)
            .rename_axis("label")
            .reset_index(name="count")
            .to_csv(OUTPUT_DIR / "news_label_counts.csv", index=False)
        )

        pd.DataFrame({
            "metric": [
                "records", "has_title_rate", "has_text_rate",
                "avg_title_len", "avg_text_len", "median_text_len",
            ],
            "value": [
                len(news_df),
                round(float(news_df["has_title"].mean()), 4),
                round(float(news_df["has_text"].mean()),  4),
                round(float(news_df["title_len"].mean()), 2),
                round(float(news_df["text_len"].mean()),  2),
                round(float(news_df["text_len"].median()), 2),
            ],
        }).to_csv(OUTPUT_DIR / "news_quality_metrics.csv", index=False)

        inv_df.to_csv(OUTPUT_DIR   / "zip_inventory.csv",           index=False)
        month_df.to_csv(OUTPUT_DIR / "articles_by_month.csv",       index=False)
        pct_df.to_csv(OUTPUT_DIR   / "text_length_percentiles.csv", index=False)

        save_plots(news_df, month_df)

    report = [
        "# FinSage Dataset EDA\n",
        f"Dataset path: `{DATASET_PATH}`\n",
    ]

    if news_df.empty:
        report.append(
            "No ZIP archives were found at the dataset path. "
            "Verify DATASET_PATH points to the `Datasets` folder of the "
            "cloned financial-news-dataset-master repository.\n"
        )
    else:
        pos_n    = int((news_df["label_name"] == "Positive").sum())
        neg_n    = int((news_df["label_name"] == "Negative").sum())
        pos_zips = int(news_df[news_df["label_name"] == "Positive"]["zip_file"].nunique())
        neg_zips = int(news_df[news_df["label_name"] == "Negative"]["zip_file"].nunique())

        report.append(
            f"Loaded **{len(news_df):,}** articles from "
            f"**{news_df['zip_file'].nunique()}** ZIP archives "
            f"({pos_zips} positive ZIPs → {pos_n:,} articles; "
            f"{neg_zips} negative ZIPs → {neg_n:,} articles).\n"
        )

        report.append("## ZIP Inventory\n")
        report.append(
            f"Date range: `{inv_df['release_date'].min()}` – "
            f"`{inv_df['release_date'].max()}`\n"
        )
        report.append(inv_df.to_markdown(index=False) + "\n")

        report.append("## Label Distribution\n")
        report.append(
            news_df["label_name"]
            .value_counts(dropna=False)
            .rename_axis("label")
            .reset_index(name="count")
            .to_markdown(index=False) + "\n"
        )

        report.append("## Monthly Article Volume\n")
        report.append(month_df.to_markdown(index=False) + "\n")

        report.append("## Schema Profile (top 20 fields by presence rate)\n")
        report.append(schema_df.head(20).to_markdown(index=False) + "\n")

        report.append("## Data Quality\n")
        report.append(pd.DataFrame({
            "metric": [
                "has_title_rate", "has_text_rate",
                "avg_title_len", "avg_text_len", "median_text_len",
            ],
            "value": [
                round(float(news_df["has_title"].mean()), 4),
                round(float(news_df["has_text"].mean()),  4),
                round(float(news_df["title_len"].mean()), 2),
                round(float(news_df["text_len"].mean()),  2),
                round(float(news_df["text_len"].median()), 2),
            ],
        }).to_markdown(index=False) + "\n")

        report.append("## Text Length Percentiles\n")
        report.append(pct_df.to_markdown(index=False) + "\n")

        report.append("## Ticker Keyword Coverage\n")
        report.append(cover_df.to_markdown(index=False) + "\n")

    report.append("## Financial Feature Template\n")
    report.append(
        "The financial anomaly detector reads six Yahoo Finance ratios live at inference time. "
        "This template records the expected feature schema for the ten baseline tickers.\n"
    )
    report.append(fin_df.to_markdown(index=False) + "\n")

    (OUTPUT_DIR / "eda_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[eda] Done. Outputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
