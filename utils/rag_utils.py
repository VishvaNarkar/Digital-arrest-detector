# utils/rag_utils.py
import requests
import json
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Ollama local endpoint (default)
OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

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

_RAG_CACHE = None

def _get_rag_entries() -> list:
    global _RAG_CACHE
    if _RAG_CACHE is not None:
        return _RAG_CACHE

    _RAG_CACHE = []
    base_dir = Path(__file__).resolve().parent.parent
    rag_dir = base_dir / "dataset" / "rag"

    if not rag_dir.exists():
        return []

    # Helper to add entries
    def add_from_file(filename, keys_to_index):
        filepath = rag_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            index_text = " ".join(str(item.get(k, "")) for k in keys_to_index)
                            rep_lines = []
                            for k, v in item.items():
                                if isinstance(v, list):
                                    v_str = ", ".join(v)
                                else:
                                    v_str = str(v)
                                rep_lines.append(f"{k.replace('_', ' ').capitalize()}: {v_str}")
                            rep = "\n".join(rep_lines)
                            
                            _RAG_CACHE.append({
                                "file": filename,
                                "index_text": index_text.lower(),
                                "representation": rep
                            })
            except Exception as e:
                logger.error(f"Failed to load RAG file {filename}: {e}")

    add_from_file("scam_knowledge.json", ["title", "description", "warning_signs", "recommended_action"])
    add_from_file("prevention_tips.json", ["title", "tips"])
    add_from_file("indian_laws.json", ["act_section", "title", "description"])
    add_from_file("police_guidelines.json", ["authority", "guideline"])
    add_from_file("fraud_patterns.json", ["pattern_name", "warning_signs", "prevention"])

    return _RAG_CACHE

def retrieve_rag_context(text: str, top_n: int = 3) -> str:
    """Retrieve top N relevant RAG documents based on keyword overlap."""
    from pathlib import Path
    entries = _get_rag_entries()
    if not entries:
        return "No local RAG context available."

    # Tokenize input text
    query_tokens = set(w.lower().strip(".,;:!?()") for w in text.split())
    query_tokens = {w for w in query_tokens if len(w) >= 3}

    scored_entries = []
    for entry in entries:
        idx_text = entry["index_text"]
        score = 0
        for token in query_tokens:
            if token in idx_text:
                score += 1
                if idx_text.startswith(token) or token in idx_text[:30]:
                    score += 2
        scored_entries.append((score, entry))

    scored_entries.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, entry in scored_entries:
        if score > 0 and len(results) < top_n:
            results.append(entry["representation"])

    # Fallback if no overlap matches
    if not results:
        for entry in entries:
            if entry["file"] in ["prevention_tips.json", "police_guidelines.json"] and len(results) < top_n:
                results.append(entry["representation"])

    context_str = ""
    for i, res in enumerate(results, 1):
        context_str += f"\n--- Context Document {i} ---\n{res}\n"
    return context_str


def rag_verify_text(text: str,
                    ml_prob: float,
                    keywords: list,
                    sentiment: dict,
                    model: str = DEFAULT_MODEL) -> Dict:
    """
    Send structured prompt to LLM to get a verification + explanation.
    Retrieves local RAG documents to supply context for the LLM.
    Returns dict with keys:
        - llm_prob (0..1)
        - verdict (Likely Scam / Likely Safe)
        - explanation (str)
        - raw (original response)
    """
    from pathlib import Path
    
    # Retrieve local RAG context
    rag_context = retrieve_rag_context(text)
    
    # Compose a clear instruction for model
    keywords_str = ", ".join(keywords) if keywords else "None"
    prompt = f"""
You are FraudShield Assistant, a specialized AI cybersecurity agent. Analyze the following message for digital arrest threats, courier parcel fraud, utility bill scams, and credential phishing.

Message to analyze:
\"\"\"{text}\"\"\"

Relevant Reference Knowledge (RAG Context):
\"\"\"{rag_context}\"\"\"

Metadata from heuristic and ML models:
- ML model probability (spam/phishing): {ml_prob:.3f}
- Detected flags & keywords: {keywords_str}
- Sentiment scores: {json.dumps(sentiment)}

Analysis Guidelines:
1. Brand Mismatch / Phishing Links: If the text mentions a brand (like HDFC, SBI, FedEx, Netflix) but the link is suspicious or doesn't match the official brand domain, it is a high-risk phishing scam.
2. High-Stakes Threats: If the text threatens "digital arrest" by authorities (CBI, Police, Customs) or utility disconnection (power bill cutoff), it is a high-risk scam (assign 90%-100% risk).
3. Requests for Credentials: If the text requests sensitive actions (sharing OTP, password, PIN, KYC details) via a link, it is a phishing scam (assign 90%-100% risk).
4. Safe/Transactional Alerts: Standard bank alerts that do NOT ask you to click a link or call an unofficial number are low-risk (Likely Safe).

Task:
1. Provide a clear classification: "Likely Scam" or "Likely Safe".
2. Give a numeric estimate of risk as a percent (0 to 100).
3. Provide a concise explanation (1-2 sentences) explaining the key reasons (e.g. presence of brand impersonation links, urgency, or digital arrest intimidation).
4. Suggest 2 short, direct, actionable advice lines for the user.

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
