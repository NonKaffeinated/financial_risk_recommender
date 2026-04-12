# sentiment.py
# News sentiment analysis using NLP on recent headlines

# Planned methodology:
#   Baseline : TF-IDF + Logistic Regression
#   Advanced : Transformer-based embeddings, sentence embeddings

# Output format 
# sentiment_analyze(ticker) must return this shape:

# {
#     "score":     float,     # -1.0 (very negative) to 1.0 (very positive)
#     "label":     str,       # "Positive", "Neutral", or "Negative"
#     "headlines": list[str]  # top 5 most relevant headlines used for analysis
# }

# Example return value:
# {
#     "score": -0.43,
#     "label": "Negative",
#     "headlines": [
#         "Company X faces regulatory scrutiny",
#         "Earnings miss expectations for third quarter"
#     ]

def sentiment_analyze(ticker: str) -> dict:
    raise NotImplementedError("sentiment.py is not implemented yet.")