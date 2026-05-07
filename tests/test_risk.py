# test_risk.py
# Run from project root: python tests/test_risk.py

from financial import financial_analyze
from risk import compute_risk
from sentiment import sentiment_analyze

tickers = ["NVDA", "AAPL", "BYND"]

for ticker in tickers:
    print(f"\n{'='*40}")
    print(f"Testing: {ticker}")
    print('='*40)

    fin    = financial_analyze(ticker)
    sent   = sentiment_analyze(ticker)
    result = compute_risk(fin, sent)

    print(f"Anomaly Score  : {fin['anomaly_score']}")
    print(f"Sentiment      : {sent['score']} ({sent['label']})")
    print(f"Trust Score    : {result['trust_score']} / 100")
    print(f"Level          : {result['level']}")
    print(f"Recommendation : {result['recommendation']}")