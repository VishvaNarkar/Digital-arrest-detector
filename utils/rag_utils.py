# utils/rag_utils.py
import requests
import json
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Ollama local endpoint (default)
OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"

# Basic health check for local Ollama instance
def is_ollama_available(timeout: float = 2.0) -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=timeout)
        return r.status_code in (200, 404)
    except Exception:
        return False

def query_ollama(prompt: str,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = 512,
                 temperature: float = 0.0,
                 stop: Optional[list] = None,
                 timeout: float = 120.0) -> Dict:
    """
    Handle Ollama's streaming JSON responses line-by-line.
    Concatenate the 'response' fields until done:true.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True  # <- key fix
    }
    if stop:
        payload["stop"] = stop

    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(OLLAMA_API, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            text_output = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if "response" in j:
                        text_output += j["response"]
                    if j.get("done"):
                        break
                except Exception:
                    continue
            return {"text": text_output}
    except Exception as e:
        logger.exception("Ollama stream query failed: %s", e)
        raise


# ---------- Prompting helpers ----------

def rag_verify_text(text: str,
                    ml_prob: float,
                    keywords: list,
                    sentiment: dict,
                    model: str = DEFAULT_MODEL) -> Dict:
    """
    Send structured prompt to LLM to get a verification + explanation.
    Returns dict with keys:
        - llm_prob (0..1)
        - verdict (Likely Scam / Likely Safe)
        - explanation (str)
        - raw (original response)
    """
    # Compose a clear instruction for model
    keywords_str = ", ".join(keywords) if keywords else "None"
    prompt = f"""
You are FraudShield Assistant. Given the following data, decide if the message is a scam and explain concisely.

Message:
\"\"\"{text}\"\"\"

Metadata:
- ML model probability (spam/phishing): {ml_prob:.3f}
- Detected risky keywords: {keywords_str}
- Sentiment scores: {json.dumps(sentiment)}

Task:
1) Provide a clear classification in one line: "Likely Scam" or "Likely Safe".
2) Give a short numeric estimate of risk as a percent (0-100).
3) Give a concise explanation (1-2 sentences) describing the main reasons (keywords, urgency, suspicious phrasing).
4) Suggest 2 short actionable advice lines for a user.

Output JSON exactly with keys: verdict, risk_percent, explanation, advice (list of strings).
"""

    try:
        resp = query_ollama(prompt, model=model, max_tokens=256, temperature=0.0)
        # Try to extract JSON from response. Many Ollama setups provide `resp["text"]`
        text_out = ""
        if isinstance(resp, dict):
            # If Ollama returned structured JSON (some local builds), try main fields:
            if "text" in resp:
                text_out = resp["text"]
            else:
                # If resp contains "choices" or similar, try to join
                for k in ("result", "output", "choices"):
                    if k in resp:
                        text_out = json.dumps(resp[k])
                        break
                if not text_out:
                    text_out = json.dumps(resp)
        else:
            text_out = str(resp)

        # Try to find JSON block inside text_out
        # Many models will output JSON; attempt to extract first {...}
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text_out[start:end+1]
            try:
                parsed = json.loads(candidate)
                # Normalize expected fields
                verdict = parsed.get("verdict") or parsed.get("classification") or parsed.get("label")
                risk = parsed.get("risk_percent") or parsed.get("risk") or parsed.get("prob_percent")
                explanation = parsed.get("explanation") or parsed.get("reason") or parsed.get("explain")
                advice = parsed.get("advice") or parsed.get("recommendations") or parsed.get("tips") or []
                if isinstance(advice, str):
                    advice = [advice]
                llm_prob = None
                try:
                    if risk is not None:
                        llm_prob = float(risk) / 100.0
                except Exception:
                    llm_prob = None
                return {
                    "llm_prob": llm_prob,
                    "verdict": verdict,
                    "risk_percent": risk,
                    "explanation": explanation,
                    "advice": advice,
                    "raw": parsed
                }
            except json.JSONDecodeError:
                # not JSON, fallthrough
                pass

        # Fallback: return raw text as explanation
        return {
            "llm_prob": None,
            "verdict": None,
            "risk_percent": None,
            "explanation": text_out.strip(),
            "advice": [],
            "raw": text_out
        }

    except Exception as e:
        logger.exception("RAG verification failed: %s", e)
        return {
            "llm_prob": None,
            "verdict": None,
            "risk_percent": None,
            "explanation": f"Ollama error: {e}",
            "advice": [],
            "raw": None
        }


def rag_refine_transcription(raw_text: str,
                             model: str = DEFAULT_MODEL) -> Dict:
    """
    Ask the model to 'clean' the raw transcript: fix punctuation, repeated tokens, incomplete words.
    Returns dict with keys:
      - cleaned_text
      - notes (any warnings)
      - raw (LLM output)
    """
    prompt = f"""
You are a helpful assistant that refines automatic transcriptions.

Input transcription (do not invent new facts; only improve readability):
\"\"\"{raw_text}\"\"\"

Tasks:
1) Fix punctuation and capitalization.
2) Correct obvious transcription artifacts (repeated words, common misrecognitions).
3) Keep the original meaning; do not add or remove factual content.
4) If you are unsure about a word, mark it with [unclear].

Return a JSON object with keys: cleaned_text, notes.
"""
    try:
        resp = query_ollama(prompt, model=model, max_tokens=512, temperature=0.0)
        text_out = ""
        if isinstance(resp, dict) and "text" in resp:
            text_out = resp["text"]
        elif isinstance(resp, str):
            text_out = resp
        else:
            text_out = json.dumps(resp)

        # try to parse JSON block
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text_out[start:end+1]
            try:
                parsed = json.loads(candidate)
                return {
                    "cleaned_text": parsed.get("cleaned_text") or "",
                    "notes": parsed.get("notes") or "",
                    "raw": parsed
                }
            except json.JSONDecodeError:
                pass

        # fallback: return text as cleaned_text
        return {"cleaned_text": text_out.strip(), "notes": "", "raw": text_out}

    except Exception as e:
        logger.exception("RAG transcription refine failed: %s", e)
        return {"cleaned_text": raw_text, "notes": f"ollama error: {e}", "raw": None}
