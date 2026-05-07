# risk.py
# Combines anomaly detection from financial reports (financial.py) and news sentiment (sentiment.py) into a composite risk score

# Methodology:
#   Baseline : Rule-based filtering
#   Advanced : Content-based filtering

# Output format
# compute_risk(financial, sentiment) must return this shape:

# {
#     "level":          str,   # "Low", "Medium", or "High"
#     "trust_score":    float, # 0 (least trustworthy) to 100 (most trustworthy)
#     "recommendation": str    # one sentence action recommendation
# }

# Input parameters:
#   financial : dict returned by financial.analyze()
#               keys: anomaly_score (float), flags (list[str]), summary (str)
#
#   sentiment : dict returned by sentiment.analyze()
#               keys: score (float), label (str), headlines (list[str])

# Example return value:
# {
#     "level": "High",
#     "trust_score": 21.5,
#     "recommendation": "Proceed with caution — multiple financial and sentiment red flags detected."
# }
import numpy as np
 
# Weights — adjust as a team based on data quality
FINANCIAL_WEIGHT = 0.85    # financial data is quarterly verified — more reliable
SENTIMENT_WEIGHT = 0.15   # news sentiment is supplementary
 
# Known company profiles for content-based filtering (advanced) 
# Feature vector: [anomaly_score, flag_count_normalized, sentiment_score_normalized]
# Expand this dataset as you gather more real results
KNOWN_PROFILES = {
    "safe": [
        [0.26, 0.0, 0.8],   # AAPL-like
        [0.28, 0.0, 0.7],   # TSLA-like
        [0.29, 0.0, 0.75],  # NVDA-like
    ],
    "risky": [
        [0.43, 0.33, 0.2],  # BYND-like
        [0.37, 0.17, 0.1],  # BBBY-like
        [0.37, 0.17, 0.15], # RIDE-like
    ]
}
 
# Baseline — rule-based 
def _rule_based(financial: dict, sentiment: dict) -> float:
    """
    Combine anomaly score and sentiment score using fixed weights.
    Returns a raw combined risk score 0-1.
    """
    anomaly    = financial.get("anomaly_score", 0)
    sent_score = sentiment.get("score", 0)
 
    # Normalize sentiment from -1->1 to 0->1 (higher = more risky)
    sent_risk  = (1 - sent_score) / 2
    combined   = (anomaly * FINANCIAL_WEIGHT) + (sent_risk * SENTIMENT_WEIGHT)
    return round(combined, 3)
 
 
# Advanced — content-based filtering 
def _content_based(financial: dict, sentiment: dict) -> float:
    """
    Compare company profile against known safe/risky profiles
    using cosine similarity. Returns a raw combined risk score 0-1.
    """
    from sklearn.metrics.pairwise import cosine_similarity
 
    anomaly    = financial.get("anomaly_score", 0)
    flags      = financial.get("flags", [])
    sent_score = sentiment.get("score", 0)
 
    # Build feature vector
    flag_norm  = min(len(flags) / 6, 1.0)
    sent_norm  = (sent_score + 1) / 2  # normalize -1->1 to 0->1
 
    vector     = np.array([[anomaly, flag_norm, sent_norm]])
 
    # Compare against known profiles
    safe_vecs  = np.array(KNOWN_PROFILES["safe"])
    risky_vecs = np.array(KNOWN_PROFILES["risky"])
 
    sim_safe   = cosine_similarity(vector, safe_vecs).mean()
    sim_risky  = cosine_similarity(vector, risky_vecs).mean()
 
    # Risk score — higher similarity to risky = higher risk
    total      = sim_safe + sim_risky
    risk_score = round(sim_risky / total if total > 0 else 0.5, 3)
    return risk_score
 
 
# Score to label conversion 
def _score_to_trust(risk_score: float) -> float:
    """Convert 0-1 risk score to 0-100 trust score (inverse)."""
    return round((1 - risk_score) * 100, 1)
 
 
def _get_level(trust_score: float) -> str:
    if trust_score >= 65:
        return "Low"
    elif trust_score >= 40:
        return "Medium"
    else:
        return "High"
 
 
def _get_recommendation(level: str, flags: list, sentiment_label: str) -> str:
    """Generate a plain English recommendation from risk signals."""
    flag_text = f" Key financial concerns: {', '.join(flags)}." if flags else ""
    sent_text = f" News sentiment is {sentiment_label.lower()}." if sentiment_label else ""
 
    if level == "Low":
        return f"Low risk — generally safe to engage.{sent_text}"
    elif level == "Medium":
        return f"Proceed with caution — some risk signals detected.{flag_text}{sent_text}"
    else:
        return f"High risk — significant red flags detected.{flag_text}{sent_text}"
 
 
 
def compute_risk(financial: dict, sentiment: dict, method: str = "rule_based") -> dict:
    """
    Combine financial and sentiment signals into a composite risk score.
 
    Parameters
    ----------
    financial : dict from financial.analyze()
                { anomaly_score: float, flags: list[str], summary: str }
    sentiment : dict from sentiment.analyze()
                { score: float, label: str, news_summary: str }
    method    : "rule_based" (baseline) or "content_based" (advanced)
 
    Returns
    -------
    {
        "level":          str,   # "Low", "Medium", or "High"
        "trust_score":    float, # 0 (least trustworthy) to 100 (most trustworthy)
        "recommendation": str    # plain English action recommendation
    }
    """
    # Step 1: compute raw risk score
    if method == "content_based":
        risk_score = _content_based(financial, sentiment)
    else:
        risk_score = _rule_based(financial, sentiment)
 
    # Step 2: convert to trust score
    trust_score = _score_to_trust(risk_score)
    trust_score = max(0.0, min(100.0, trust_score))  # clamp 0-100
 
    # Step 3: determine level
    level = _get_level(trust_score)
 
    # Step 4: generate recommendation
    recommendation = _get_recommendation(
        level,
        financial.get("flags", []),
        sentiment.get("label", ""),
    )
 
    return {
        "level":          level,
        "trust_score":    trust_score,
        "recommendation": recommendation,
    }

    #raise NotImplementedError("risk.py is not implemented yet.")