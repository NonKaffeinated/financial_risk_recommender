# Streamlit application

"""
app.py — Streamlit UI & control layer for Financial Risk & Trustworthiness Chatbot
Integrates: chatbot.py (ME), financial.py (teammate), sentiment.py (teammate), risk.py (teammate)
"""

import os
import streamlit as st
import time

SCORING_MARKER = "### Data & scores used\n\n"
SCORING_REPLY_SEP = "\n\n---\n\n"

# Checks for a scoring section
def _split_scoring_reply(content: str) -> tuple[str | None, str]:
    """If this assistant turn includes a scoring section, return (scoring_md, reply); else (None, full)."""
    if content.startswith(SCORING_MARKER) and SCORING_REPLY_SEP in content:
        body = content[len(SCORING_MARKER) :]
        scoring, reply = body.split(SCORING_REPLY_SEP, 1)
        return scoring.strip(), reply
    return None, content


def write_introduction():
    intro = f"""
Welcome to FinSage, your trusted advisor for financial risk and trustworthiness!
Ask about any company. FinSage breaks down financial health risk and news sentiment into a clear picture.

Here's what I can help with:
- **Financial health** — anomaly detection, key ratios, and red flags
- **Risk identification** — spot potential threats to a company's financial stability
- **News sentiment** — understand public and media sentiment around any company

What company can FinSage help you analyze today? Use the format {{Company/Ticker}}.
    """.strip().split(' ')

    for word in intro:
        yield word + " "
        time.sleep(0.02)  # Simulate typing effect

# Page config (must be first Streamlit call)
st.set_page_config(
    page_title="FinSage · Financial Risk Recommender",
    page_icon="💰",
    layout="centered",
)

# Load modules with visible startup spinners
with st.spinner("Training Isolation Forest on baseline companies..."):
    try:
        from financial import financial_analyze
        FINANCIAL_READY = True
    except ImportError:
        FINANCIAL_READY = False
        def financial_analyze(ticker: str) -> dict:
            return {
                "anomaly_score": None,
                "flags":         [],
                "summary":       "Financial module not available.",
                "ratios":        {}
            }

with st.spinner("Training Logistic Regression and TF-IDF on financial news articles..."):
    try:
        from sentiment import sentiment_analyze
        SENTIMENT_READY = True
    except ImportError:
        SENTIMENT_READY = False
        def sentiment_analyze(ticker: str) -> dict:
            return {
                "score":        None,
                "label":        "N/A",
                "news_summary": "Sentiment module not available.",
            }

with st.spinner("Weighting risk scores..."):
    try:
        from risk import compute_risk
        RISK_READY = True
    except ImportError:
        RISK_READY = False
        def compute_risk(financial: dict, sentiment: dict, method: str = "rule_based") -> dict:
            return {
                "level":          "N/A",
                "trust_score":    None,
                "recommendation": "Awaiting full module integration.",
                "raw_risk_score": None,
                "method":         method,
            }

with st.spinner("Loading Llama 3 model from Ollama..."):
    from chatbot import chat, extract_ticker, format_scoring_markdown

ESERVER_READY = False
_eserver = None

if "eserver_initialized" not in st.session_state:
    with st.spinner("Loading earnings RAG server..."):
        try:
            from earning_server import EServer
            _eserver = EServer()
            ESERVER_READY = True
            st.session_state.eserver_ready = True
            st.session_state._eserver = _eserver
            print("[app] EServer ready.")
        except Exception as e:
            ESERVER_READY = False
            _eserver = None
            st.session_state.eserver_ready = False
            st.session_state._eserver = None
            print(f"[app] EServer not available: {e}")
    st.session_state.eserver_initialized = True
else:
    ESERVER_READY = st.session_state.eserver_ready
    _eserver = st.session_state._eserver

# Header
st.title("FinSage")
st.caption("Financial Risk Recommender · Powered by Ollama (local)")
st.divider()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

# Sidebar
with st.sidebar:
    # Header
    st.title("Main Menu")
    st.subheader("Welcome to FinSage!")
    st.divider()

    def _pill(label, ready):
        dot = '<span style="color:#4ade80">●</span>' if ready else '<span style="color:#f87171">●</span>'
        status = "Ready" if ready else "Not Ready"
        return f'<span>{dot} {label} — {status}</span>'
    
    # Display module availability and display status
    st.markdown(
    _pill("financial.py",      FINANCIAL_READY) + "<br>" +
    _pill("sentiment.py",      SENTIMENT_READY) + "<br>" +
    _pill("risk.py",           RISK_READY)      + "<br>" +
    _pill("earning_server.py", ESERVER_READY),
    unsafe_allow_html=True,
)

    # Clear chat button
    if st.button("Clear chat"):
        st.session_state.chat_history = [] # Reset chat history
        st.session_state.intro_shown = False # Reset intro flag
        st.rerun() 
    
# If intro not shown, show it and save to history. Otherwise, render chat history from session state. 
# Ensures introduction is shown only on first visit or after clearing chat, and chat history persists across interactions without re-rendering the intro.
if not st.session_state.intro_shown:
    with st.chat_message("assistant"):
        intro = st.write_stream(write_introduction())
    st.session_state.chat_history.append({"role": "assistant", "content": intro})
    st.session_state.intro_shown = True
else:    
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                scoring_body, reply_body = _split_scoring_reply(message["content"])
                if scoring_body is not None:
                    with st.expander("Data & scores used for this answer", expanded=False):
                        st.markdown(scoring_body)
                    st.markdown(reply_body)
                else:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])


# User input
if prompt := st.chat_input("Ask about a company's financial risk..."):

    ticker = extract_ticker(prompt)
    earnings_rec = None

    risk_method = os.environ.get("RISK_METHOD", "rule_based")

    if ticker is not None:
        with st.spinner(f"Loading data for {ticker}..."):
            try:
                fin = financial_analyze(ticker)
                sent = sentiment_analyze(ticker)
                risk = compute_risk(fin, sent, method=risk_method)

                earnings_rec = None
                if ESERVER_READY and _eserver:
                    try:
                        full_query = f"Should an individual investor buy, sell, or hold {ticker} stock before the next earnings date?"
                        earnings_rec = _eserver.query_RAG_str(full_query)
                        if not earnings_rec:
                            print(f"[app] No Pinecone earnings docs found for {ticker}; skipping RAG portion.")
                            earnings_rec = None
                    except Exception as e:
                        print(f"[app] EServer query failed: {e}")

                scoring_md = format_scoring_markdown(ticker, fin, sent, risk, earnings_rec)
            except Exception as exc:
                st.warning(f"Could not load scoring data for {ticker}: {exc}")
                scoring_md = f'Could not load scoring data for {ticker}: {exc}'
            prompt = f"{prompt} + \n\n\nREPORT FOR '{ticker}':\n{scoring_md}"

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt}) 

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.write_stream(
                chat(st.session_state.chat_history) # history includes current turn
            )
    # Append assistant reply to history
    st.session_state.chat_history.append({"role": "assistant", "content": reply})  # save for next turn