# financial.py
# Fraud & anomaly detection using financial data from Yahoo Finance / SEC

# Methodology
#   MVP      : Isolation Forest (yfinance used, unsupervised, no labeled data needed) ← done 
#   Baseline : Logistic Regression, Random Forest (needs labeled SEC fraud dataset)
#   Advanced : XGBoost

# Output format
# financial_analyze(ticker) must return this shape:
#
# {
#     "anomaly_score": float,    # 0.0 (normal) to 1.0 (highly anomalous)
#     "flags":         list[str],# list of specific warning signs found
#     "summary":       str       # one paragraph description of financial health
# }

# Example return value:
# {
#     "anomaly_score": 0.72,
#     "flags": ["negative profit margins", "high debt-to-equity ratio"],
#     "summary": "The company shows signs of financial stress with declining margins."

# Setup:
# pip install yfinance scikit-learn numpy

import os
import numpy as np
import yfinance as yf
from sklearn.ensemble import IsolationForest

# Config
BASELINE_TICKERS = [
    "AAPL", "MSFT", "JNJ", "JPM", "PG",
    "V",    "UNH",  "HD",  "MA",  "DIS",
]

# Feature Extraction - Pre-train a simple anomaly detection model on baseline tickers
def _extract_features(ticker: str) -> list:
    """
    Pull key financial ratios from Yahoo Finance.
    Returns a feature vector for anomaly detection.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return [0.0] * 6
 
    return [
        float(info.get("debtToEquity",     0) or 0),
        float(info.get("profitMargins",    0) or 0),
        float(info.get("revenueGrowth",    0) or 0),
        float(info.get("currentRatio",     0) or 0),
        float(info.get("returnOnEquity",   0) or 0),
        float(info.get("operatingMargins", 0) or 0),
    ]
 
 
def _build_flags(ticker: str) -> list:
    """
    Check financial ratios against thresholds and return warning flags.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return ["Could not retrieve financial data"]
 
    flags = []
 
    if (info.get("debtToEquity")     or 0) > 200:
        flags.append("high debt-to-equity ratio")
    if (info.get("profitMargins")    or 0) < 0:
        flags.append("negative profit margins")
    if (info.get("revenueGrowth")    or 0) < 0:
        flags.append("negative revenue growth")
    if (info.get("currentRatio")     or 0) < 1:
        flags.append("current ratio below 1 — liquidity risk")
    if (info.get("returnOnEquity")   or 0) < 0:
        flags.append("negative return on equity")
    if (info.get("operatingMargins") or 0) < 0:
        flags.append("negative operating margins")
 
    return flags
 
 
def _build_summary(ticker: str, flags: list, anomaly_score: float) -> str:
    """
    Generate a plain English summary combining company info and risk flags.
    """
    try:
        info        = yf.Ticker(ticker).info
        company     = info.get("longName", ticker)
        description = info.get("longBusinessSummary", "")[:300]
    except Exception:
        company     = ticker
        description = ""
 
    if not flags:
        risk_text = f"{company} shows no significant financial anomalies."
    else:
        level     = "high" if anomaly_score > 0.6 else "moderate"
        flag_str  = ", ".join(flags)
        risk_text = f"{company} shows {level} financial risk. Key concerns: {flag_str}."
 
    return f"{risk_text} {description}...".strip()

# Isolation Forest 
def _train_isolation_forest() -> IsolationForest:
    """
    Train Isolation Forest on a basket of healthy S&P 500 companies.
    This defines what 'normal' looks like financially.
    """
    print("[financial] Training Isolation Forest on baseline companies...")
    baseline_features = []
 
    for t in BASELINE_TICKERS:
        try:
            features = _extract_features(t)
            baseline_features.append(features)
        except Exception:
            continue
 
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(baseline_features)
    print("[financial] Model ready.")
    return model

_model = _train_isolation_forest()

# Analyze financial health for a given ticker
def financial_analyze(ticker: str) -> dict:
    """
    Analyze financial anomalies for a given ticker.
 
    Parameters
    ----------
    ticker : company ticker symbol e.g. "NVDA", "AAPL"
 
    Returns
    -------
    {
        "anomaly_score": float,     # 0.0 (normal) to 1.0 (highly anomalous)
        "flags":         list[str], # list of specific warning signs
        "summary":       str        # plain English financial health summary
    }
    """
    # Step 1: extract features
    features = _extract_features(ticker)
 
    # Step 2: score with Isolation Forest
    # decision_function: more negative = more anomalous
    raw_score = _model.decision_function([features])[0]
    iso_score = round((1 - raw_score) / 2, 2)
    iso_score = max(0.0, min(1.0, iso_score))

    # Step 3: build flags from thresholds
    flags = _build_flags(ticker)

    # Step 4: blend isolation forest + flag count for better spread
    flag_score    = min(len(flags) / 6, 1.0)           # normalize flags 0-1
    anomaly_score = round((iso_score * 0.6) + (flag_score * 0.4), 2)
    anomaly_score = max(0.0, min(1.0, anomaly_score))  # clamp 0-1

    # Step 5: build summary
    summary = _build_summary(ticker, flags, anomaly_score)
 
    return {
        "anomaly_score": anomaly_score,
        "flags":         flags,
        "summary":       summary,
    }

    #raise NotImplementedError("financial.py is not implemented yet.")