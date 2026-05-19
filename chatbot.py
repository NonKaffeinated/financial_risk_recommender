# Turns models into human-readable responses
# Handles all natural-language Q&A, explanation generation, and chat history
import re
import requests
import json
import os
import yfinance as yf
from risk import FINANCIAL_WEIGHT, SENTIMENT_WEIGHT

# Make sure Ollama is running: https://ollama.com
# Then pull a model: ollama pull llama3 
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """
You are FinSage, a financial risk and trustworthiness analyst AI.
You explain financial risk signals and news sentiment in clear, plain language for finance enthusiasts.

When the user message includes a `---` section with "Structured analysis" (the pipeline output):
- Base your answer entirely on that block: trust score, risk level, anomaly score, sentiment label/score, and recommendation text.
- Restate and explain only those metrics. Do not introduce other scores (e.g. "out of 5", alternate composite scores), revenue figures, cash figures, or ratings not in the block.
- If something is missing from the block, say it is not in this snapshot rather than inventing it.

When given a structured risk report for a company you:
1. Summarise the overall risk level in 1-2 sentences (match the risk level and trust score from the data).
2. Highlight the top signals (financial anomalies, sentiment, composite/trust score as given).
3. Explain what each signal means in plain English.
4. Give a brief recommendation aligned with the recommendation line in the data.
5. Answer follow-up questions about the report with precision.

When no grounding / structured analysis is provided:
- Answer general questions about financial risk and investing
- Explain what metrics matter when evaluating a company
- Discuss industries and market trends in general

Always be concise, factual, and professional.
Never fabricate specific scores, percentages, ratings, or data sources.
Only use data explicitly provided to you and never invent numbers.
""".strip()

_TICKER_BLOCKLIST = frozenset({
    "HOLD", "SELL", "BUY", "STOCKS", "STOCK", "SHOULD",
    "NEWS", "WHY", "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
    "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD",
    "SEE", "TWO", "WHO", "BOY", "DID", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "LOW",
    "HIGH", "RISK", "FROM", "WHAT", "WITH", "HAVE", "THIS", "THAT", "WILL", "YOUR", "ANY",
    "WHEN", "DOES", "THAN", "THEM", "VERY", "INTO", "JUST", "OVER", "ALSO", "ONLY",
    "STOCK", "STOCKS", "NYSE", "NASDAQ", "ETF", "IPO", "CEO", "CFO", "EPS", "DCF", "YOY",
    "MOST", "SOME", "MANY", "SUCH", "ABOUT", "AFTER", "BACK", "BEEN", "CALL", "CAME",
    "COME", "EACH", "EVEN", "FIND", "FIRST", "GIVE", "GOOD", "GREAT", "HAND", "HERE",
    "KEEP", "KNOW", "LAST", "LEFT", "LIFE", "LIKE", "LONG", "LOOK", "MADE", "MAKE", "MUCH",
    "MUST", "NAME", "NEED", "NEXT", "OPEN", "PART", "REAL", "RIGHT", "SAID", "SAME", "SEEM",
    "SHOW", "SIDE", "TAKE", "TELL", "THESE", "THEY", "THINK", "TIME", "UNDER", "WANT", "WAYS",
    "WELL", "WENT", "WERE", "WORK", "YEAR", "HELP", "SELL", "BUY", "HOLD", "CASH", "DEBT",
    "IS", "IT", "AS", "AN", "OR", "BE", "AT", "SO", "NO", "ON", "IN", "OF", "TO", "UP",
    "IF", "OK", "DO", "GO", "ME", "MY", "WE", "US", "AM", "PM",
})

def _resolve_ticker(query: str) -> str | None:
    """Resolve a company name or ticker hint to a valid equity symbol via yfinance."""
    query = query.strip()
    if not query or query.upper() in _TICKER_BLOCKLIST:
        return None

    try:
        results = yf.Search(query, max_results=5)
        quotes = results.quotes or []
    except Exception:
        quotes = []

    if quotes:
        want = query.upper()
        for quote in quotes:
            if quote.get("quoteType") == "EQUITY" and quote.get("symbol", "").upper() == want:
                return quote.get("symbol")
        for quote in quotes:
            if quote.get("quoteType") == "EQUITY":
                return quote.get("symbol")

    return _validate_ticker_direct(query)


def _validate_ticker_direct(query: str) -> str | None:
    """Fallback when search fails: accept input if Yahoo recognizes it as an equity."""
    symbol = query.upper()
    if symbol in _TICKER_BLOCKLIST:
        return None
    try:
        info = yf.Ticker(symbol).info
        if info.get("quoteType") == "EQUITY":
            return info.get("symbol") or symbol
    except Exception:
        pass
    return None


def extract_ticker(prompt: str) -> str | None:
    """
    Extract {company or ticker} from the prompt and resolve to a Yahoo equity symbol.
    Example: "How risky is {Nvidia}?" -> "NVDA"
    """
    matches = re.findall(r"\{([^{}]+)\}", prompt)
    if not matches:
        return None
    return _resolve_ticker(matches[0])

def format_scoring_markdown(
    ticker: str,
    financial: dict,
    sentiment: dict,
    risk: dict,
) -> str:
    return "\n".join(
        [
            f"**Ticker:** `{ticker}`",
            "",
            "**Financial signals** (Yahoo Finance + Isolation Forest anomaly blend)",
            f"- Anomaly score (0 = normal, 1 = highly anomalous): **{financial.get('anomaly_score')}**",
            f"- Flags: {', '.join(financial.get('flags') or ['(none)'])}",
            f"- Summary: {financial.get('summary', '')}",
            "",
            "**News sentiment** (baseline model on matched articles, if available)",
            f"- Score (−1 to 1): **{sentiment.get('score')}**",
            f"- Label: **{sentiment.get('label')}**",
            f"- Summary: {sentiment.get('news_summary', '')}",
            "",
            "**Composite risk** (rule-based blend: "
            f"{FINANCIAL_WEIGHT:.0%} financial + {SENTIMENT_WEIGHT:.0%} sentiment → raw risk 0–1, then trust score)",
            f"- Method: **{risk.get('method', 'rule_based')}**",
            f"- Raw risk score (0–1): **{risk.get('raw_risk_score')}**",
            f"- Trust score (0–100): **{risk.get('trust_score')}**",
            f"- Risk level: **{risk.get('level')}**",
            f"- Recommendation: {risk.get('recommendation', '')}",
        ]
    )

def chat(
    history: list[dict]
):

    """
    Stream tokens for st.write_stream.

    *history* must be the conversation so far **not including** the current user turn
    (the app stores the user message in session state separately).
    """
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    OLLAMA_MODEL,
            "messages": history,
            "stream":   True,        
        },
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    full_reply = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            full_reply += token
            yield token        

