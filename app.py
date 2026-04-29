# Streamlit application

"""
app.py — Streamlit UI & control layer for Financial Risk & Trustworthiness Chatbot
Integrates: chatbot.py (ME), financial.py (teammate), sentiment.py (teammate), risk.py (teammate)
"""

from numpy import dot
import streamlit as st
from chatbot import chat
import time


def write_introduction():
    intro = """
Welcome to FinSage, your trusted advisor for financial risk and trustworthiness!
Ask about any company. FinSage breaks down financial health risk and news sentiment into a clear picture.

Here's what I can help with:
- **Financial health** — anomaly detection, key ratios, and red flags
- **Risk identification** — spot potential threats to a company's financial stability
- **News sentiment** — understand public and media sentiment around any company

What company can FinSage help you analyze today?
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

# Header
st.title("FinSage")
st.caption("Financial Risk Recommender · Powered by Ollama (local)")
st.divider()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

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
            st.markdown(message["content"])

# Check module availability and set flags
try:
    from financial import analyze as financial_analyze
    FINANCIAL_READY = True
except ImportError:
    FINANCIAL_READY = False

try:
    from sentiment import analyze as sentiment_analyze
    SENTIMENT_READY = True
except ImportError:
    SENTIMENT_READY = False

try:
    from risk import evaluate as risk_assess
    RISK_READY = True
except ImportError:
    RISK_READY = False

# Sidebar
with st.sidebar:
    # Header
    st.title("Main Menu")
    st.subheader("Welcome to FinSage!")
    st.divider()

    st.subheader("Module Status")
    def _pill(label, ready):
        dot = '<span style="color:#4ade80">●</span>' if ready else '<span style="color:#f87171">●</span>'
        status = "ready" if ready else "pending"
        return f'<span>{dot} {label} — {status}</span>'

    st.markdown(
        _pill("financial.py", FINANCIAL_READY) + "<br>" +
        _pill("sentiment.py", SENTIMENT_READY) + "<br>" +
        _pill("risk.py",      RISK_READY),
        unsafe_allow_html=True,
    )

    # Clear chat button
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.intro_shown = False # Reset intro flag
        st.rerun() 

# User input
if prompt := st.chat_input("Ask about a company's financial risk..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # st.session_state.chat_history.append({"role": "user", "content": prompt}) 
    # No need to save user message to history, implicitly done by streamlit

    # Stream assistant response in one bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.write_stream(chat(prompt, st.session_state.chat_history))

    # Save completed reply to history
    st.session_state.chat_history.append({"role": "assistant", "content": reply})