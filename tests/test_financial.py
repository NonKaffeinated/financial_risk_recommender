# test_financial.py
# Run from project root: python tests/test_financial.py

from financial import financial_analyze

# Test with a few tickers
tickers = ["NVDA", "AAPL", "TSLA"]

for ticker in tickers:
    print(f"\n{'='*40}")
    print(f"Testing: {ticker}")
    print('='*40)
    
    result = financial_analyze(ticker)
    
    print(f"Anomaly Score : {result['anomaly_score']}")
    print(f"Flags         : {result['flags']}")
    print(f"Summary       : {result['summary'][:200]}...")

risky_tickers = ["BBBY", "BYND", "RIDE"]  # historically struggling companies

for ticker in risky_tickers:
    print(f"\n{'='*40}")
    print(f"Testing: {ticker}")
    print('='*40)
    
    result = financial_analyze(ticker)
    
    print(f"Anomaly Score : {result['anomaly_score']}")
    print(f"Flags         : {result['flags']}")
    print(f"Summary       : {result['summary'][:200]}...")