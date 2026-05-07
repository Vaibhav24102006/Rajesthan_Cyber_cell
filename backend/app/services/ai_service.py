import json
import logging
import os
import re
from typing import Any, Dict

import requests
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)


CRIME_TYPES = [
    "Unknown",
    "Financial Fraud",
    "Phishing",
    "Hacking",
    "Harassment",
    "Identity Theft",
    "Ransomware",
    "Social Media Crime",
    "Cyber Stalking",
    "Other",
]

DEFAULT_SEVERITY = 4
DEFAULT_CRIME_TYPE = "Unknown"
DEFAULT_SUMMARY = "No summary generated"


def _heuristic_fallback(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    crime_type = "Other"

    if any(k in lowered for k in ["otp", "upi", "transaction", "account", "money", "rupee", "withdrawn"]):
        crime_type = "Financial Fraud"
    elif any(k in lowered for k in ["phishing", "fake link", "link", "email"]):
        crime_type = "Phishing"
    elif any(k in lowered for k in ["hack", "hacked", "breach", "unauthorized access"]):
        crime_type = "Hacking"
    elif any(k in lowered for k in ["blackmail", "harass", "abuse", "threat"]):
        crime_type = "Harassment"
    elif any(k in lowered for k in ["aadhaar", "identity", "kyc", "pan card"]):
        crime_type = "Identity Theft"
    elif any(k in lowered for k in ["ransom", "bitcoin", "encrypted"]):
        crime_type = "Ransomware"

    severity = 7 if crime_type in {"Financial Fraud", "Identity Theft", "Ransomware"} else 5
    summary = "Complaint indicates possible cyber offense; officer should verify extracted details and proceed with legal workflow."

    return {"crime_type": crime_type, "severity": severity, "summary": summary}


def _sanitize_json_text(raw_text: str) -> str:
    cleaned = raw_text.strip().replace("```json", "").replace("```", "")
    return cleaned


def _extract_json_block(raw_text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        raise ValueError("No JSON object found in model response")
    return match.group(0)


def _parse_model_json(raw_text: str) -> Dict[str, Any]:
    cleaned = _sanitize_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_block = _extract_json_block(cleaned)
        return json.loads(json_block)


def _normalize_text_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return None
    text_value = str(value).strip()
    if text_value == "" or text_value.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return None
    return text_value


def _normalize_severity(value: Any) -> int:
    if value is None:
        return DEFAULT_SEVERITY
    try:
        severity_num = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return DEFAULT_SEVERITY
    return max(1, min(severity_num, 10))


def _normalize_crime_type(value: Any) -> str:
    normalized = _normalize_text_field(value)
    if not normalized:
        return DEFAULT_CRIME_TYPE
    for crime in CRIME_TYPES:
        if normalized.lower() == crime.lower():
            return crime
    return DEFAULT_CRIME_TYPE


def _normalize_summary(value: Any) -> str:
    normalized = _normalize_text_field(value)
    if not normalized:
        return DEFAULT_SUMMARY
    return normalized


def _normalize_ai_data(ai_data: Any) -> Dict[str, Any]:
    # Handles null / malformed / unexpected model payloads defensively.
    if not isinstance(ai_data, dict):
        ai_data = {}

    normalized: Dict[str, Any] = {
        "victim_name": _normalize_text_field(ai_data.get("victim_name")),
        "phone": _normalize_text_field(ai_data.get("phone")),
        "location": _normalize_text_field(ai_data.get("location")),
        "amount": _normalize_text_field(ai_data.get("amount")),
        "transaction_id": _normalize_text_field(ai_data.get("transaction_id")),
        "account_number": _normalize_text_field(ai_data.get("account_number")),
        "bank": _normalize_text_field(ai_data.get("bank")),
        "crime_type": _normalize_crime_type(ai_data.get("crime_type")),
        "severity": _normalize_severity(ai_data.get("severity")),
        "summary": _normalize_summary(ai_data.get("summary")),
    }
    return normalized


def _ask_ollama(prompt: str, ollama_url: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    response = requests.post(ollama_url, json=payload, timeout=90)
    response.raise_for_status()
    body = response.json()
    text_payload = body.get("response")
    if not text_payload:
        raise ValueError("Ollama returned empty response")
    return text_payload


def enrich_with_ai(text: str, regex_data: Dict[str, Any]) -> Dict[str, Any]:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    model_name = os.getenv("OLLAMA_MODEL", "phi3")
    prompt = f"""
You are an extraction and classification assistant for Jaipur Cyber Cell.
Input may be Hindi, English, or Hinglish.

Regex extracted fields:
{json.dumps(regex_data, ensure_ascii=False)}

Complaint text:
\"\"\"{text}\"\"\"

Return ONLY valid JSON object in this schema:
{{
  "victim_name": string|null,
  "phone": string|null,
  "location": string|null,
  "amount": string|null,
  "transaction_id": string|null,
  "account_number": string|null,
  "bank": string|null,
  "crime_type": one of {CRIME_TYPES},
  "severity": integer 1-10,
  "summary": string (max 25 words)
}}
Do not include markdown or commentary.
"""
    try:
        text_payload = _ask_ollama(prompt, ollama_url, model_name)
        ai_data = _parse_model_json(text_payload)
    except Exception as first_exc:
        logger.warning("Primary Ollama parse failed: %s. Retrying with stricter prompt.", first_exc)
        retry_prompt = (
            prompt
            + "\n\nSTRICT OUTPUT RULE: return exactly one JSON object only. No explanation, no markdown."
        )
        try:
            retry_payload = _ask_ollama(retry_prompt, ollama_url, model_name)
            ai_data = _parse_model_json(retry_payload)
        except Exception as second_exc:
            logger.error("Ollama extraction failed after retry: %s", second_exc)
            ai_data = _heuristic_fallback(text)

    safe_ai_data = _normalize_ai_data(ai_data)
    return safe_ai_data
