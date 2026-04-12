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

def compute_risk(financial: dict, sentiment: dict) -> dict:
    raise NotImplementedError("risk.py is not implemented yet.")