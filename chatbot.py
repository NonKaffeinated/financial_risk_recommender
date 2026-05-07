# Turns models into human-readable responses
# Handles all natural-language Q&A, explanation generation, and chat history
import requests
import json
import os

# Make sure Ollama is running: https://ollama.com
# Then pull a model: ollama pull llama3 
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """
You are FinSage, a financial risk and trustworthiness analyst AI.
You explain financial risk signals and news sentiment in clear, plain language for finance enthusiasts.

When given a structured risk report for a company you:
1. Summarise the overall risk level in 1-2 sentences.s
2. Highlight the top signals (financial anomalies, sentiment, composite score).
3. Explain what each signal means in plain English.
4. Give a brief recommendation based on the data provided.
5. Answer follow-up questions about the report with precision.

When no risk report is provided you can:
- Answer general questions about financial risk and investing
- Explain what metrics matter when evaluating a company
- Discuss industries and market trends in general

Always be concise, factual, and professional. 
Never fabricate specific scores, percentages, ratings, or data sources.
Only use data explicitly provided to you and never invent numbers.
""".strip()

def chat(user_message: str, history: list[dict]):
    """Generator — yields tokens for st.write_stream."""
    history.append({"role": "user", "content": user_message})

    all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

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
