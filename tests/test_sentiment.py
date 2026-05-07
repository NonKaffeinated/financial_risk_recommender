# test_sentiment.py
# Run from project root: python tests/test_sentiment.py

from sentiment import sentiment_analyze

tickers = ["NVDA", "AAPL", "TSLA", "BYND"]

for ticker in tickers:
    print(f"\n{'='*40}")
    print(f"Testing: {ticker}")
    print('='*40)

    result = sentiment_analyze(ticker)

    print(f"Score        : {result['score']}")
    print(f"Label        : {result['label']}")
    print(f"News Summary : {result['news_summary'][:200]}...")