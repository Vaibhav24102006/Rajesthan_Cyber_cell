import logging
import os
from typing import Dict, Any, Tuple
import requests

logger = logging.getLogger(__name__)

# Prepopulated translation cache for common cybercrime regional/Hinglish expressions
TRANSLATION_CACHE = {
    "mera bank khata se paise kat gaya": "Money was deducted from my bank account",
    "mera bank account se paise nikal gaye": "Money was withdrawn from my bank account",
    "galti se click ho gaya": "Clicked on it by mistake",
    "link open kiya": "Opened the phishing link",
    "upi pin share kiya": "Shared the UPI PIN",
    "paise cut gaye": "Money was deducted",
    "mharo paisa kat gayo": "My money was deducted",
    "telegram par contact kiya": "Contacted via Telegram",
    "whatsapp par bol raha tha": "Was speaking on WhatsApp",
    "link par open kiya": "Opened the link",
    "mohe link bhejo aur mero bank khato se paiso kat gayo": "A link was sent to me and money was deducted from my bank account",
    "mujhe link bhejo aur mera bank khata se paise kat gaya": "A link was sent to me and money was deducted from my bank account",
}

def translate_text(
    text: str,
    detected_dialect: str,
    reconstruction_mode: str = "structured_english_output",
    extracted_entities: Dict[str, Any] = None
) -> Tuple[str, float]:
    """
    Pass 4 Ingestion & Translation:
    Translates standard Hindi or Romanized Hinglish to readable standard English.
    Skipped if reconstruction_mode is preserve_original_language or normalized_hindi.
    """
    if not text or not text.strip():
        return "", 1.0

    # 1. Skip translation if requested by mode
    if reconstruction_mode in ["preserve_original_language", "normalized_hindi"]:
        logger.info("Translation skipped due to reconstruction mode: %s", reconstruction_mode)
        return text, 1.0

    # 2. Check in-memory translation cache (exact/sub-phrase lookup)
    text_lower = text.strip().lower().rstrip(".,?!")
    if text_lower in TRANSLATION_CACHE:
        logger.info("Translation cache hit!")
        return TRANSLATION_CACHE[text_lower], 0.99

    # Check for sub-phrase mappings
    for raw_phrase, eng_translation in TRANSLATION_CACHE.items():
        if raw_phrase in text_lower and len(text_lower) < len(raw_phrase) + 10:
            logger.info("Translation partial cache hit!")
            return eng_translation, 0.95

    # 3. LLM-based Translation
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    model_name = os.getenv("OCR_CORRECTION_MODEL", "phi3")

    prompt = f"""You are a certified professional translator for the Rajasthan Cyber Cell.
Translate the following normalized Hindi/Hinglish complaint text into clean, formal, standard investigative English.

STRICT TRANSLATION RULES:
1. Translate all Hindi sentences and Hinglish expressions into correct English.
2. Keep the meaning and chronological narrative EXACTLY as reported. Do not add or guess information.
3. Keep all names, transactions, bank names, and numerical values intact. Do not translate proper names.
4. Strictly protect these critical fields:
{f"- Extracted Entities: {extracted_entities}" if extracted_entities else "None"}
5. Output ONLY the translated English text. Do not add explanations, conversational filler, or markdown.

Normalized Text:
\"\"\"
{text}
\"\"\"
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
        },
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=20)
        if response.ok:
            translated_result = response.json().get("response", "").strip()
            if translated_result:
                # Add to cache for future requests
                TRANSLATION_CACHE[text_lower] = translated_result
                return translated_result, 0.88
    except Exception as exc:
        logger.error("Translation LLM execution failed: %s. Returning untranslated text.", exc)

    return text, 0.50
