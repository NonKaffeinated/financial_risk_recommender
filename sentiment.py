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
#     "news_summary": str     # News summary: one paragraph plain English summary of the news sentiment and key headlines
# }

# Example return value:
# {
#     "score": -0.43,
#     "label": "Negative",
#     "headlines": [
#         "Company X faces regulatory scrutiny",
#         "Earnings miss expectations for third quarter"
#     ]
# Setup:
#   pip install scikit-learn sentence-transformers pinecone-client
#
# Dataset:
#   git clone https://github.com/Webhose/financial-news-dataset
#   Set DATASET_PATH below to the cloned folder path
 
import os
import json
import glob
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
 
# Config
DATASET_PATH = os.environ.get("DATASET_PATH", "financial-news-dataset/Datasets/extracted/")
USE_PINECONE   = os.environ.get("USE_PINECONE", "false").lower() == "true"
PINECONE_KEY   = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "financial-news")
 
# Common company name mappings (ticker → name for keyword search)
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
 
LABEL_MAP = {1: "Positive", 0: "Negative", 2: "Neutral"}
 
# Startup: train model once on load 
_vectorizer = None
_model      = None
_articles   = []
 
def _startup():
    """Load dataset and train baseline model on startup."""
    global _vectorizer, _model, _articles
 
    if USE_PINECONE:
        print("[sentiment] Using Pinecone knowledge base.")
        return
 
    print(f"[sentiment] Loading Webhose dataset from {DATASET_PATH} ...")
    _articles = _load_articles(DATASET_PATH)
 
    if not _articles:
        print("[sentiment] No articles found. Check DATASET_PATH.")
        return
 
    print(f"[sentiment] Loaded {len(_articles)} articles. Training model ...")
    _vectorizer, _model = _train_model(_articles)
    print("[sentiment] Model ready.")
 
 
def _load_articles(path: str) -> list:
    articles = []
    for filepath in glob.glob(os.path.join(path, "**/*.json"), recursive=True):
        # Extract label from filename
        if "positive" in filepath.lower():
            label = 1
        elif "negative" in filepath.lower():
            label = 0
        else:
            label = 2  # neutral

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                article = json.load(f)
                article["_label"] = label  # attach label to article
                articles.append(article)
            except json.JSONDecodeError:
                continue
    return articles

def _train_model(articles: list):
    texts  = [a.get("text", a.get("title", ""))[:500] for a in articles]
    labels = [a.get("_label", 2) for a in articles]  # ← real labels, not predicted

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X          = vectorizer.fit_transform(texts)
    model      = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model
 

# Article fetching 
 
def _get_articles_local(ticker: str) -> list:
    """Filter local Webhose articles by ticker or company name."""
    keyword  = TICKER_TO_NAME.get(ticker.upper(), ticker.lower())
    relevant = []
    for article in _articles:
        text = (article.get("text", "") + " " + article.get("title", "")).lower()
        if any(k in text for k in keyword.split()):
            relevant.append(article)
    return relevant
 
 
def _get_articles_pinecone(ticker: str) -> list:
    """
    Query Pinecone knowledge base for articles related to the ticker.
    Swap in when knowledge base is ready.
 
    Requirements:
    - PINECONE_API_KEY set in .env
    - PINECONE_INDEX set in .env
    - Must confirm:
        1. Embedding model used (must match _embed() below)
        2. Metadata fields stored (ticker, text, title, date)
    """
    from pinecone import Pinecone
 
    pc    = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(PINECONE_INDEX)
 
    embedding = _embed(ticker)
    results   = index.query(
        vector=embedding,
        top_k=10,
        include_metadata=True,
        filter={"ticker": {"$eq": ticker.upper()}}
    )
    return [match["metadata"] for match in results.get("matches", [])]
 
 
def _embed(text: str) -> list:
    """
    Convert text to embedding vector.
    Must match the embedding model used to build the Pinecone index.
    Default: all-MiniLM-L6-v2
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text).tolist()
 
 
# Scoring
def _score_baseline(articles: list) -> tuple:
    if _vectorizer is None or _model is None:
        return 0.0, "Neutral" # fallback if model not ready
 
    texts  = [a.get("text", a.get("title", ""))[:500] for a in articles[:10]]
    X      = _vectorizer.transform(texts)
    preds  = _model.predict(X)
    probs  = _model.predict_proba(X)
 
    scores = []
    for pred, prob in zip(preds, probs):
        if pred == 1:   scores.append(prob[1])
        elif pred == 0: scores.append(-prob[0])
        else:           scores.append(0.0)
 
    avg_score = round(sum(scores) / len(scores), 3)
    label     = "Positive" if avg_score > 0.1 else "Negative" if avg_score < -0.1 else "Neutral"
    return avg_score, label
 
def _score_finbert(articles: list) -> tuple:
    """
    Advanced: score using FinBERT (financial-specific transformer).
    Swap in for _score_baseline when ready.
    pip install transformers torch
    """
    from transformers import pipeline
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
 
    scores = []
    for article in articles[:10]:
        text   = article.get("text", article.get("title", ""))[:512]
        result = finbert(text)[0]
        score  = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        scores.append(score)
 
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    label     = "Positive" if avg_score > 0.1 else "Negative" if avg_score < -0.1 else "Neutral"
    return avg_score, label
 
 
def _build_summary(articles: list, score: float, ticker: str) -> str:
    tone    = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    excerpt = articles[0].get("text", articles[0].get("title", ""))[:300] if articles else ""
    return (
        f"News sentiment for {ticker} is {tone} based on "
        f"{len(articles)} articles. Most relevant excerpt: {excerpt}..."
    )

# Analyze news sentiment for a given ticker
def sentiment_analyze(ticker: str) -> dict:
    """
    Analyze news sentiment for a given ticker.
 
    Parameters
    ----------
    ticker : company ticker symbol e.g. "NVDA", "AAPL"
 
    Returns
    -------
    {
        "score":        float,  # -1.0 to 1.0
        "label":        str,    # "Positive", "Neutral", "Negative"
        "news_summary": str     # plain English summary of news findings
    }
    """
    # Step 1: fetch relevant articles
    if USE_PINECONE:
        articles = _get_articles_pinecone(ticker)
    else:
        articles = _get_articles_local(ticker)
 
    if not articles:
        return {
            "score":        0.0,
            "label":        "Neutral",
            "news_summary": f"No news articles found for {ticker}.",
        }
 
    # Step 2: score — swap _score_baseline for _score_finbert for advanced
    score, label = _score_baseline(articles)
 
    # Step 3: build summary
    news_summary = _build_summary(articles, score, ticker)
 
    return {
        "score":        score,
        "label":        label,
        "news_summary": news_summary,
    }

    #raise NotImplementedError("sentiment.py is not implemented yet.")

# Run startup 
_startup()