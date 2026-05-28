#benchmark


!pip install vaderSentiment

import os
import re
import json
import glob
import zipfile
import importlib.util
import time
from pathlib import Path



import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)




def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load financial.py and risk.py by explicit file path so the benchmark works
# regardless of the working directory or sys.path configuration.
try:
    _fin_mod = _load_module("financial", "/content/drive/MyDrive/finsage/financial.py")
    financial_analyze = _fin_mod.financial_analyze
    _FINANCIAL_AVAILABLE = True
except Exception as _fin_err:
    _FINANCIAL_AVAILABLE = False
    print(f"[benchmark] financial.py unavailable: {_fin_err}")

try:
    _risk_mod = _load_module("risk", "/content/drive/MyDrive/finsage/risk.py")
    compute_risk = _risk_mod.compute_risk
    _RISK_AVAILABLE = True
except Exception as _risk_err:
    _RISK_AVAILABLE = False
    print(f"[benchmark] risk.py unavailable: {_risk_err}")

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/content/drive/MyDrive/financial-news-dataset-master/Datasets",
)
OUTPUT_DIR = Path("/content/drive/MyDrive/financial-news-dataset-master/output_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

# Labeled ticker set for financial and risk benchmarks.
# Healthy tickers are held out from the Isolation Forest training set
# (training set: AAPL, MSFT, JNJ, JPM, PG, V, UNH, HD, MA, DIS).

FINANCIAL_TEST_SET = {
    # healthy — add 3 more well-known stable tickers
    "NVDA": 0, "GOOGL": 0, "AMZN": 0, "META": 0, "COST": 0,
    "TSLA": 0, "ADBE": 0, "CRM": 0,
    # risky/distressed — add 3 more known troubled tickers
    "BYND": 1, "AMC":  1, "NKLA": 1, "LCID": 1, "BLNK": 1,
    "BBBY": 1, "RIDE": 1, "SPCE": 1,
}
EXPECTED_RISK_LEVEL = {
    "NVDA": "Low",  "GOOGL": "Low",  "AMZN": "Low",  "META": "Low",  "COST": "Low",
    "BYND": "High", "AMC":   "High", "NKLA": "High", "LCID": "High", "BLNK": "High",
}

# Neutral sentinel used when isolating the risk aggregation logic from sentiment.
NEUTRAL_SENTIMENT = {"score": 0.0, "label": "Neutral", "news_summary": ""}

ANOMALY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _label_from_zip(zip_name: str) -> int:
    low = zip_name.lower()
    if "positive" in low:
        return 1
    if "negative" in low:
        return 0
    return -1


def load_news_articles(path: str) -> pd.DataFrame:
    rows = []
    for zip_path in sorted(glob.glob(os.path.join(path, "*.zip"))):
        basename = os.path.basename(zip_path)
        label = _label_from_zip(basename)
        if label == -1:
            continue
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
                        text = a.get("text", "") or a.get("title", "") or ""
                        rows.append({"label": label, "text": text})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared metrics helper
# ---------------------------------------------------------------------------

def _build_metrics(name: str, y_true, y_pred, y_proba=None) -> dict:
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    row = {
        "model":              name,
        "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_binary":   round(float(prec_b), 4),
        "recall_binary":      round(float(rec_b),  4),
        "f1_binary":          round(float(f1_b),   4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted":    round(float(rec_w),  4),
        "f1_weighted":        round(float(f1_w),   4),
    }
    if y_proba is not None:
        try:
            row["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            row["roc_auc"] = np.nan
    else:
        row["roc_auc"] = np.nan
    return row

def benchmark_risk_model(financial_cache: dict, news_df: pd.DataFrame = None) -> tuple:
    ...
    # existing neutral-sentiment pass (already there)
    for ticker, expected_level in EXPECTED_RISK_LEVEL.items():
        fin = financial_cache.get(ticker, {})
        rb  = compute_risk(fin, NEUTRAL_SENTIMENT, method="rule_based")
        ...

    # NEW: real-sentiment pass if news data available
    if news_df is not None and not news_df.empty:
        sia = SentimentIntensityAnalyzer()
        for ticker, expected_level in EXPECTED_RISK_LEVEL.items():
            keyword  = ticker.lower()
            mask     = news_df["text"].str.lower().str.contains(keyword, regex=False)
            subset   = news_df[mask]["text"].head(50)
            score    = float(np.mean([sia.polarity_scores(t)["compound"] for t in subset])) if len(subset) else 0.0
            real_sentiment = {"score": score, "label": "Positive" if score > 0 else "Negative"}
            rb_real  = compute_risk(financial_cache[ticker], real_sentiment, method="rule_based")
            # append to rows with a "real_sentiment" column flag

# ---------------------------------------------------------------------------
# Benchmark 1 — Sentiment models (VADER, TF-IDF variants)
# ---------------------------------------------------------------------------

def benchmark_sentiment_models(news_df: pd.DataFrame) -> tuple:
    if news_df.empty or news_df["label"].nunique() < 2:
        return pd.DataFrame([{
            "model": "all", "status": "skipped",
            "reason": "dataset unavailable or single class present",
        }]), {}

    df = news_df[news_df["label"].isin([0, 1])].copy().reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"],
    )

    results = []
    cms = {}

    # VADER zero-shot baseline (compound threshold per VADER paper)
    print("[benchmark] Running VADER ...")
    sia = SentimentIntensityAnalyzer()
    vader_preds = [
        1 if sia.polarity_scores(t)["compound"] >= 0.05 else 0
        for t in X_test.tolist()
    ]
    results.append(_build_metrics("vader_baseline", y_test, vader_preds))
    cms["vader_baseline"] = confusion_matrix(y_test, vader_preds)

    # TF-IDF vectorizers
    vec_prod = TfidfVectorizer(max_features=5000, stop_words="english")
    Xtr_prod = vec_prod.fit_transform(X_train.str[:500])
    Xte_prod = vec_prod.transform(X_test.str[:500])

    vec_full = TfidfVectorizer(max_features=5000, stop_words="english")
    Xtr_full = vec_full.fit_transform(X_train)
    Xte_full = vec_full.transform(X_test)

    model_configs = [
        (
            "tfidf_logreg_production",
            LogisticRegression(max_iter=1000),
            Xtr_prod, Xte_prod,
        ),
        (
            "tfidf_logreg_balanced",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            Xtr_full, Xte_full,
        ),
        (
            "tfidf_rf_balanced",
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
            Xtr_full, Xte_full,
        ),
    ]

    for name, model, Xtr, Xte in model_configs:
        print(f"[benchmark] Training {name} ...")
        model.fit(Xtr, y_train)
        pred  = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None
        results.append(_build_metrics(name, y_test, pred, proba))
        cms[name] = confusion_matrix(y_test, pred)

    return pd.DataFrame(results), cms


def save_confusion_matrices(cms: dict) -> None:
    rows = []
    for name, cm in cms.items():
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            rows.append({
                "model":          name,
                "true_negative":  int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive":  int(tp),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)


# ---------------------------------------------------------------------------
# Benchmark 2 — 5-fold cross-validation for LogReg models (Gap 4 fix)
# Pipeline prevents data leakage between folds.
# ---------------------------------------------------------------------------

def cross_validate_logreg(news_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    if news_df.empty or news_df["label"].nunique() < 2:
        return pd.DataFrame()

    df = news_df[news_df["label"].isin([0, 1])].copy().reset_index(drop=True)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    configs = [
        (
            "tfidf_logreg_production_cv",
            Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("clf",   LogisticRegression(max_iter=1000)),
            ]),
            df["text"].str[:500],
        ),
        (
            "tfidf_logreg_balanced_cv",
            Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]),
            df["text"],
        ),
    ]

    rows = []
    for name, pipe, X in configs:
        print(f"[benchmark] {n_splits}-fold CV for {name} ...")
        scores = cross_validate(
            pipe, X, df["label"],
            cv=cv,
            scoring=["accuracy", "f1_weighted", "roc_auc"],
            n_jobs=-1,
        )
        rows.append({
            "model":            name,
            "cv_folds":         n_splits,
            "mean_accuracy":    round(float(scores["test_accuracy"].mean()),    4),
            "std_accuracy":     round(float(scores["test_accuracy"].std()),     4),
            "mean_f1_weighted": round(float(scores["test_f1_weighted"].mean()), 4),
            "std_f1_weighted":  round(float(scores["test_f1_weighted"].std()),  4),
            "mean_roc_auc":     round(float(scores["test_roc_auc"].mean()),     4),
            "std_roc_auc":      round(float(scores["test_roc_auc"].std()),      4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark 3 — Financial anomaly detection model (Gap 1 fix)
# Fetches live Yahoo Finance data for labeled healthy / risky tickers.
# Returns the raw financial results for reuse in the risk benchmark.
# ---------------------------------------------------------------------------

def _fetch_financial_results() -> dict:
    if not _FINANCIAL_AVAILABLE:
        return {}
    cache = {}
    for ticker in FINANCIAL_TEST_SET:
        print(f"[benchmark] financial_analyze({ticker}) ...")
        cache[ticker] = financial_analyze(ticker)
    return cache


def benchmark_financial_model(financial_cache: dict) -> tuple:
    if not financial_cache:
        return pd.DataFrame([{
            "ticker": "all", "status": "skipped",
            "reason": "financial.py unavailable or no results fetched",
        }]), {}

    rows = []
    for ticker, true_label in FINANCIAL_TEST_SET.items():
        result = financial_cache.get(ticker, {})
        anomaly = result.get("anomaly_score", np.nan)
        pred_label = 1 if (anomaly is not np.nan and anomaly >= ANOMALY_THRESHOLD) else 0
        rows.append({
            "ticker":          ticker,
            "true_label":      true_label,
            "anomaly_score":   anomaly,
            "flag_count":      len(result.get("flags", [])),
            "predicted_label": pred_label,
            "correct":         int(pred_label == true_label),
        })

    df = pd.DataFrame(rows)
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    summary = {
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 3),
        "precision":  round(float(prec), 3),
        "recall":     round(float(rec),  3),
        "f1":         round(float(f1),   3),
        "n_tickers":  len(df),
        "threshold":  ANOMALY_THRESHOLD,
    }

    return df, summary


# ---------------------------------------------------------------------------
# Benchmark 4 — Composite risk module (Gap 2 fix)
# Runs both rule_based and content_based methods against labeled tickers.
# Uses neutral sentiment to isolate the financial aggregation logic.
# ---------------------------------------------------------------------------

def benchmark_risk_model(financial_cache: dict) -> tuple:
    if not financial_cache or not _RISK_AVAILABLE:
        return pd.DataFrame([{
            "ticker": "all", "status": "skipped",
            "reason": "financial.py or risk.py unavailable",
        }]), pd.DataFrame()

    rows = []
    for ticker, expected_level in EXPECTED_RISK_LEVEL.items():
        fin = financial_cache.get(ticker, {})
        rb  = compute_risk(fin, NEUTRAL_SENTIMENT, method="rule_based")
        cb  = compute_risk(fin, NEUTRAL_SENTIMENT, method="content_based")
        rows.append({
            "ticker":                ticker,
            "expected_level":        expected_level,
            "rule_based_level":      rb["level"],
            "rule_based_trust":      rb["trust_score"],
            "rule_based_correct":    int(rb["level"] == expected_level),
            "content_based_level":   cb["level"],
            "content_based_trust":   cb["trust_score"],
            "content_based_correct": int(cb["level"] == expected_level),
        })

    df = pd.DataFrame(rows)

    healthy = df[df["expected_level"] == "Low"]
    risky   = df[df["expected_level"] == "High"]

    # Spearman: expected trust (Low→80, High→20) vs actual trust score
    expected_trust = df["expected_level"].map({"Low": 80, "High": 20})

    rb_spearman, _ = spearmanr(expected_trust, df["rule_based_trust"])
    cb_spearman, _ = spearmanr(expected_trust, df["content_based_trust"])

    summary = pd.DataFrame([
        {
            "method":             "rule_based",
            "accuracy":           round(float(df["rule_based_correct"].mean()),     3),
            "spearman_r":         round(float(rb_spearman),                          3),
            "mean_trust_healthy": round(float(healthy["rule_based_trust"].mean()),  1),
            "mean_trust_risky":   round(float(risky["rule_based_trust"].mean()),    1),
        },
        {
            "method":             "content_based",
            "accuracy":           round(float(df["content_based_correct"].mean()),  3),
            "spearman_r":         round(float(cb_spearman),                          3),
            "mean_trust_healthy": round(float(healthy["content_based_trust"].mean()), 1),
            "mean_trust_risky":   round(float(risky["content_based_trust"].mean()),   1),
        },
    ])

    return df, summary


# ---------------------------------------------------------------------------
# Benchmark 5 — FinBERT vs TF-IDF (Gap 3 fix)
# Samples a stratified subset to keep inference time manageable.
# FinBERT is zero-shot so sampling from the full dataset is valid.
# ---------------------------------------------------------------------------

def benchmark_finbert(news_df: pd.DataFrame, n_sample: int = 10000) -> dict:
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return {
            "model": "finbert_prosusai", "status": "skipped",
            "reason": "transformers not installed",
        }

    df = news_df[news_df["label"].isin([0, 1])].copy()
    if df.empty:
        return {
            "model": "finbert_prosusai", "status": "skipped",
            "reason": "no labeled articles available",
        }

    per_class = n_sample // 2
    parts = []
    for label in [0, 1]:
        sub = df[df["label"] == label]
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=42))
    sample_df = pd.concat(parts).reset_index(drop=True)

    print(
        f"[benchmark] Running FinBERT on {len(sample_df)} articles "
        "(this will take several minutes) ..."
    )

    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512,
        batch_size=16,
    )

    start = time.time()
    raw_results = pipe(sample_df["text"].str[:512].tolist())
    elapsed = time.time() - start

    # FinBERT returns "positive" / "negative" / "neutral".
    # Map "positive" → 1, anything else → 0 for binary evaluation.
    preds = [1 if r["label"] == "positive" else 0 for r in raw_results]

    metrics = _build_metrics("finbert_prosusai", sample_df["label"], preds)
    metrics["articles_sampled"] = len(sample_df)
    metrics["inference_seconds"] = round(elapsed, 1)
    metrics["sec_per_article"]   = round(elapsed / len(sample_df), 3)
    return metrics


# ---------------------------------------------------------------------------
# Remaining evaluation gaps (post-implementation)
# ---------------------------------------------------------------------------

def evaluation_gaps() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "module":  "financial.py",
            "implemented": "10-ticker labeled benchmark with anomaly_score threshold at 0.5",
            "remaining_limitation": (
                "Small sample (10 tickers) limits statistical power; "
                "yfinance returns current live data rather than point-in-time "
                "historical snapshots, so results for recovering firms may not "
                "reflect their distressed state"
            ),
        },
        {
            "module":  "risk.py",
            "implemented": "Per-ticker rule_based vs content_based comparison with accuracy and Spearman r",
            "remaining_limitation": (
                "Neutral sentiment proxy isolates financial signals but does not "
                "test the full sentiment-weighted blend; content_based method "
                "relies on only 6 hardcoded reference profiles, which limits "
                "its generalization to new company types"
            ),
        },
        {
            "module":  "sentiment.py (FinBERT path)",
            "implemented": "Side-by-side F1 and inference time comparison on a 2,000-article stratified sample",
            "remaining_limitation": (
                "2,000-article sample may underestimate variance; full-dataset "
                "FinBERT inference would require GPU acceleration to be practical; "
                "FinBERT neutral-to-negative mapping may suppress recall for "
                "ambiguous articles"
            ),
        },
        {
            "module":  "benchmark_models.py",
            "implemented": "5-fold stratified CV for both LogReg variants via sklearn Pipeline",
            "remaining_limitation": (
                "Random Forest evaluated on a single split only (5-fold CV "
                "with 300 trees on 83K documents would take 30+ minutes); "
                "no temporal split to test model drift on newer articles"
            ),
        },
    ])

def cross_validate_rf(news_df: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    if news_df.empty or news_df["label"].nunique() < 2:
        return pd.DataFrame()
    df = news_df[news_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf",   RandomForestClassifier(n_estimators=300, random_state=42,
                                         n_jobs=-1, class_weight="balanced")),
    ])
    scores = cross_validate(pipe, df["text"], df["label"], cv=cv,
                            scoring=["accuracy", "f1_weighted", "roc_auc"], n_jobs=-1)
    return pd.DataFrame([{
        "model":            "tfidf_rf_balanced_cv",
        "cv_folds":         n_splits,
        "mean_accuracy":    round(float(scores["test_accuracy"].mean()),    4),
        "std_accuracy":     round(float(scores["test_accuracy"].std()),     4),
        "mean_f1_weighted": round(float(scores["test_f1_weighted"].mean()), 4),
        "std_f1_weighted":  round(float(scores["test_f1_weighted"].std()),  4),
        "mean_roc_auc":     round(float(scores["test_roc_auc"].mean()),     4),
        "std_roc_auc":      round(float(scores["test_roc_auc"].std()),      4),
    }])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[benchmark] Loading articles from {DATASET_PATH} ...")
    news_df = load_news_articles(DATASET_PATH)

    # --- Sentiment benchmarks ---
    sentiment_df, cms = benchmark_sentiment_models(news_df)
    sentiment_df.to_csv(OUTPUT_DIR / "sentiment_benchmark_results.csv", index=False)
    if cms:
        save_confusion_matrices(cms)

    # --- Cross-validation ---
    cv_df = cross_validate_logreg(news_df)
    if not cv_df.empty:
        cv_df.to_csv(OUTPUT_DIR / "logreg_cv_results.csv", index=False)

    # --- FinBERT ---
    finbert_result = benchmark_finbert(news_df)
    finbert_df = pd.DataFrame([finbert_result])
    finbert_df.to_csv(OUTPUT_DIR / "finbert_benchmark_results.csv", index=False)

    # --- Financial model (fetches yfinance once, reused by risk benchmark) ---
    financial_cache = _fetch_financial_results()
    financial_df, financial_summary = benchmark_financial_model(financial_cache)
    financial_df.to_csv(OUTPUT_DIR / "financial_model_benchmark.csv", index=False)

    # --- Risk model ---
    risk_df, risk_summary = benchmark_risk_model(financial_cache)
    risk_df.to_csv(OUTPUT_DIR     / "risk_model_benchmark.csv",   index=False)
    risk_summary.to_csv(OUTPUT_DIR / "risk_model_summary.csv",    index=False)

    # --- Evaluation gaps ---
    gaps_df = evaluation_gaps()
    gaps_df.to_csv(OUTPUT_DIR / "model_evaluation_gaps.csv", index=False)

    # --- Report ---
    report = [
        "# FinSage Benchmark and Evaluation\n",
        f"Dataset path: `{DATASET_PATH}`\n",
    ]

    if news_df.empty:
        report.append(
            "No ZIP archives were found. Verify DATASET_PATH points to the "
            "`Datasets` folder of the cloned financial-news-dataset-master repository.\n"
        )
    else:
        pos = int((news_df["label"] == 1).sum())
        neg = int((news_df["label"] == 0).sum())
        report.append(
            f"Loaded **{len(news_df):,}** articles "
            f"({pos:,} positive, {neg:,} negative, "
            f"imbalance ratio {pos / neg:.2f}:1). "
            f"Train/test split: 80/20 stratified (random_state=42).\n"
        )

    report.append("## Sentiment Model Benchmark (single split)\n")
    report.append(
        "`tfidf_logreg_production` replicates the 500-char truncation and "
        "hyperparameters in `sentiment.py`. "
        "`tfidf_logreg_balanced` and `tfidf_rf_balanced` use full article text "
        "with `class_weight='balanced'`. "
        "VADER runs zero-shot on the test set (compound threshold ≥ 0.05).\n"
    )
    report.append(sentiment_df.to_markdown(index=False) + "\n")

    cm_path = OUTPUT_DIR / "confusion_matrices.csv"
    if cm_path.exists():
        report.append("## Confusion Matrices\n")
        report.append(pd.read_csv(cm_path).to_markdown(index=False) + "\n")

    if not cv_df.empty:
        report.append("## LogReg 5-Fold Cross-Validation\n")
        report.append(
            "Vectorizer is fitted inside each fold via `sklearn.Pipeline` "
            "to prevent data leakage.\n"
        )
        report.append(cv_df.to_markdown(index=False) + "\n")

    if finbert_result.get("status") != "skipped":
        report.append("## FinBERT Benchmark\n")
        report.append(
            f"Evaluated on a {finbert_result.get('articles_sampled', 0):,}-article "
            "stratified sample. Label mapping: FinBERT `positive` → 1, all others → 0.\n"
        )
        report.append(finbert_df.to_markdown(index=False) + "\n")
    else:
        report.append("## FinBERT Benchmark\n")
        report.append(f"Skipped: {finbert_result.get('reason', 'unknown reason')}\n")

    report.append("## Financial Anomaly Detection Benchmark\n")
    if financial_summary:
        report.append(
            f"Isolation Forest evaluated on {financial_summary['n_tickers']} labeled tickers "
            f"(anomaly_score threshold = {financial_summary['threshold']}). "
            f"Healthy tickers held out from the training set.\n"
        )
        report.append(financial_df.to_markdown(index=False) + "\n")
        report.append(
            f"**Accuracy:** {financial_summary['accuracy']} | "
            f"**Precision (risky):** {financial_summary['precision']} | "
            f"**Recall (risky):** {financial_summary['recall']} | "
            f"**F1 (risky):** {financial_summary['f1']}\n"
        )
    else:
        report.append("Skipped: financial.py unavailable.\n")

    report.append("## Composite Risk Model Benchmark\n")
    if not risk_summary.empty and "status" not in risk_df.columns:
        report.append(
            "Both methods evaluated against labeled tickers using neutral sentiment "
            "to isolate the financial aggregation logic.\n"
        )
        report.append("### Per-ticker results\n")
        report.append(risk_df.to_markdown(index=False) + "\n")
        report.append("### Method summary\n")
        report.append(risk_summary.to_markdown(index=False) + "\n")
    else:
        report.append("Skipped: financial.py or risk.py unavailable.\n")

    report.append("## Remaining Evaluation Gaps\n")
    report.append(gaps_df.to_markdown(index=False) + "\n")

    (OUTPUT_DIR / "benchmark_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[benchmark] Done. Outputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
