# Turns models into human-readable responses
# Handles all natural-language Q&A, explanation generation, and chat history
import re
import requests
import json
import os

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
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
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

# Company / brand names → Yahoo ticker (e.g. "NVIDIA!" has no NVDA symbol for the regex path)
_NAME_TO_TICKER: dict[str, str] = {
    "berkshire hathaway": "BRK-B",
    "bank of america": "BAC",
    "johnson and johnson": "JNJ",
    "jpmorgan": "JPM",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "disney": "DIS",
    "intel": "INTC",
    "amd": "AMD",
    "oracle": "ORCL",
    "cisco": "CSCO",
    "broadcom": "AVGO",
    "qualcomm": "QCOM",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "nvidia corporation": "NVDA",
    "bed bath and beyond": "BBBY",
    "beyond meat": "BYND",
}


def extract_ticker(prompt: str) -> str | None:
    """Resolve a ticker from symbols (NVDA) or company names (NVIDIA → NVDA)."""
    for m in re.finditer(r"\b([A-Z]{2,5})\b", prompt.upper()):
        token = m.group(1)
        if token not in _TICKER_BLOCKLIST:
            return token

    lower = prompt.lower()
    for name in sorted(_NAME_TO_TICKER.keys(), key=len, reverse=True):
        if " " in name:
            pattern = r"\b" + r"\s+".join(re.escape(p) for p in name.split()) + r"\b"
        else:
            pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, lower):
            return _NAME_TO_TICKER[name]

    return None


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
    user_message: str,
    history: list[dict],
    scoring_context: str | None = None,
):
    """
    Stream tokens for st.write_stream.

    *history* must be the conversation so far **not including** the current user turn
    (the app stores the user message in session state separately).
    """
    if scoring_context:
        user_content = (
            f"{user_message}\n\n---\n"
            "Structured analysis — the only numbers and labels you may cite for this company:\n"
            f"{scoring_context}"
        )
    else:
        user_content = user_message

    all_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    OLLAMA_MODEL,
            "messages": all_messages,
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
