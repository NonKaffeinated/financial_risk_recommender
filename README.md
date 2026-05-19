# FinSage - Financial Risk Recommender
A conversational AI chatbot that analyzes companies using financial signals and news sentiment to estimate risk and trustworthiness — and explains the results in plain English.

## Project Structure
```bash
project/
├── eda_benchmark_eval.ipynb        # EDA, benchmark, and evaluation of models
├── app.py                          # Streamlit UI & orchestration
├── chatbot.py                      # Conversation logic & LLM explanation
├── financial.py                    # Financial anomaly detection
├── sentiment.py                    # News sentiment analysis
├── risk.py                         # Composite risk scoring
├── tests/
│   ├── test_financial.py           # Test financial module in isolation
│   ├── test_sentiment.py           # Test sentiment module in isolation
│   └── test_risk.py                # Test full pipeline end to end
├── financial-news-dataset/         # Webhose dataset (not committed to git)
├── .env                            # API keys (not committed to git)
├── .gitignore
└── README.md
```

## Getting Started
### 1. Clone the repo
```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the news dataset
```bash
git clone https://github.com/Webhose/financial-news-dataset
cd financial-news-dataset/Datasets
unzip "*.zip" -d extracted/
cd ../..
```

### 4. Install and set up Ollama (local LLM — free, no API key)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Start Ollama (runs in background)
ollama serve
```

### 5. Set up environment variables (optional)

Create a `.env` file for optional config:
```
# LLM provider — default is ollama (free, local)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434

# Dataset path (if not in default location)
DATASET_PATH=financial-news-dataset/Datasets/extracted/

# Pinecone (when knowledge base is ready — Felix)
PINECONE_API_KEY=...
PINECONE_INDEX=financial-news
USE_PINECONE=false
```

> Never commit `.env` to GitHub. It is already in `.gitignore`.

### 6. Run the app
```bash
streamlit run app.py
```

### 7. Ask about a company using a ticker or the company name
Ask with braces: **What's the risk for {Apple}?** or **{NVDA}**

Note: yfinance resolves name/ticker to an equity symbol before analysis. Without {...}, you get general chat only (no scoring).


## Module Contract

Each module must return exactly this shape. Do not change keys or types.

| File | Function | Returns |
|---|---|---|
| `financial.py` | `financial_analyze(ticker: str)` | `{ anomaly_score: float, flags: list[str], summary: str }` |
| `sentiment.py` | `sentiment_analyze(ticker: str)` | `{ score: float, label: str, news_summary: str }` |
| `risk.py` | `compute_risk(financial: dict, sentiment: dict)` | `{ level: str, trust_score: float, recommendation: str, raw_risk_score: float, method: str }` |


## Methodology

### Financial Anomaly Detection (`financial.py`)
- **MVP:** Isolation Forest on Yahoo Finance financial ratios (unsupervised, no labeled data needed)
- **Baseline:** Logistic Regression / Random Forest with labeled SEC fraud dataset**
- **Advanced:** XGBoost**
- **Data source:** `yfinance` — no API key required
- **Features:** debt-to-equity, profit margins, revenue growth, current ratio, return on equity, operating margins

### News Sentiment Analysis (`sentiment.py`)
- **Baseline:** TF-IDF + Logistic Regression on local Webhose dataset (81,000 labeled articles)
- **Advanced:** FinBERT transformer embeddings for financial-specific NLP
- **Data source:** [Webhose financial news dataset](https://github.com/Webhose/financial-news-dataset) — downloaded locally
- **Pinecone:** ready to swap in when knowledge base is built (`USE_PINECONE=true`)
- **Embedding model:** `all-MiniLM-L6-v2` — must match Pinecone index when switching

### Composite Risk Score (`risk.py`)
- **Baseline:** Rule-based weighted combination (financial 85%, sentiment 15%)
- **Advanced:** Content-based filtering using cosine similarity against known company profiles**
- **Trust score:** 0 (least trustworthy) to 100 (most trustworthy)
- **Thresholds:** Low ≥ 65, Medium ≥ 40, High < 40

**Not yet implemented

## Progress

**MVP (Completed)**
- [x] `financial.py` — Isolation Forest on Yahoo Finance ratios
- [x] `sentiment.py` — TF-IDF + Logistic Regression on Webhose dataset
- [x] `risk.py` — Rule-based weighted trust score
- [x] `chatbot.py` — Ollama-powered conversational interface
- [x] `app.py` — Streamlit UI with streaming responses
- [x] Full pipeline tested end to end

**Future Work**
- [ ] Implement real-time news articles instead of static dataset (regularly updated)
- [ ] Improve article retrieval
- [ ] Improve Streamlit UI
- [ ] Experiment with various advanced models and parameters to improve risk analysis
- [ ] `financial.py` — Logistic Regression / Random Forest with SEC fraud dataset
- [ ] `risk.py` — Content-based filtering

## Testing

Run each module in isolation before wiring into the app:

```bash
# Test financial module
python tests/test_financial.py

# Test sentiment module
python tests/test_sentiment.py

# Test full pipeline
python tests/test_risk.py
```

Expected output for healthy companies (NVDA, AAPL):
- Anomaly score: 0.25 - 0.35
- Sentiment: Positive
- Trust score: 70+ / 100
- Level: Low

Expected output for struggling companies (BYND, BBBY):
- Anomaly score: 0.40+
- Sentiment: Neutral / Negative
- Trust score: < 65 / 100
- Level: Medium / High

## Team

| Person | Responsibility |
|---|---|
| **Kathleen** | `app.py`, `chatbot.py`, `llm.py`, `sentiment.py`, `financial.py` — UI, conversation logic, LLM integration, news sentiment analysis, Yahoo Finance anomaly detection |
| **Felix** | Pinecone knowledge base — financial news vector store |
| **Kevin** | EDA, benchmarking and evaluation|


## Important Notes

- **Dataset not in repo** — clone Webhose dataset separately and unzip into `financial-news-dataset/Datasets/extracted/`
- **Ensure Ollama llama3 is running** - refer to "Getting Started"
- **Do not change function signatures** — `app.py` and `chatbot.py` depend on them
- **Pinecone ready** — set `USE_PINECONE=true` in `.env` when Felix's knowledge base is ready, no other changes needed
- **Limitations** - Smaller companies, foreign companies, and recently listed companies may have insufficient article coverage to produce a reliable sentiment score. In these cases the system defaults to a neutral score of 0.0, which reduces the sentiment signal's contribution to the composite risk score. 