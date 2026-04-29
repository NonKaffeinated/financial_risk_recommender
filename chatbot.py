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
The full analysis feature is not yet available. When a user asks about a specific company's risk, financial health, or news sentiment, let them know 
that live data modules are coming soon and you cannot provide real scores or analysis yet. If asked for specific company data, always clarify it is not yet available.

You can still:
- Explain general concepts about financial risk
- Describe what metrics matter when evaluating a company
- Answer questions about investing and financial health in general

Always be concise, factual, and professional. Never fabricate specific scores, percentages, ratings, or data sources.
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
