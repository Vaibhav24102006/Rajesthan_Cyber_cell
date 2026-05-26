import logging
import os
from typing import Dict, Any, Tuple
import requests

logger = logging.getLogger(__name__)

# Pre-baked token mappings for Rajasthani dialects
RAJASTHANI_DICT = {
    "mharo": "mera",
    "mhara": "mere",
    "mhane": "mujhe",
    "mhanu": "mera",
    "tharo": "tera",
    "thara": "tere",
    "thane": "tujhe",
    "thanu": "tera",
    "vuno": "uska",
    "vuna": "unka",
    "kune": "kisne",
    "kathe": "kaha",
    "kathin": "kaha",
    "jathin": "jaha",
    "koni": "nahi",
    "konya": "nahi",
    "koini": "nahi",
    "nhin": "nahi",
    "nhvya": "nahi hua",
    "gayo": "gaya",
    "aayo": "aaya",
    "kat gayo": "kat gaya",
    "katgyo": "kat gaya",
    "rupiya": "rupaye",
    "rupyaji": "rupaye",
    "chhoro": "ladka",
    "chhori": "ladki",
    "lugai": "aurat",
    "laado": "beti",
    "baatan": "baate",
    "katai": "kabhi",
    "kar gayo": "kar gaya",
    "karvyo": "karwaya",
    "avya": "aaya",
    "bhado": "kiraya",
    "dhokha": "dhokha",
    "mohe": "mujhe",
    "tohe": "tumhe",
    "mero": "mera",
    "tero": "tera",
    "kahe": "kyon",
    "kachu": "kuch",
    "nahin": "nahi",
    "bhayo": "hua",
    "paiso": "paise",
    "khato": "khata",
    "thagyo": "thaga",
}

def pre_normalize_text(text: str) -> str:
    """
    Performs dictionary-based direct word replacement to clean raw dialect tokens.
    """
    words = text.split()
    normalized_words = []
    for word in words:
        # Strip punctuation for dictionary lookup
        cleaned_word = word.strip(".,;:?!'\"()[]{}").lower()
        if cleaned_word in RAJASTHANI_DICT:
            # Maintain original word casing if possible (simplistic capitalization check)
            replacement = RAJASTHANI_DICT[cleaned_word]
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Restore trailing punctuation
            suffix = word[len(cleaned_word):]
            normalized_words.append(replacement + suffix)
        else:
            normalized_words.append(word)
    return " ".join(normalized_words)

def normalize_dialect(
    text: str,
    detected_dialect: str,
    ocr_confidence: float = 1.0,
    extracted_entities: Dict[str, Any] = None
) -> Tuple[str, float]:
    """
    Pass 3 Dialect Normalization:
    Transforms Mewari, Marwari, Shekhawati or Hinglish inputs into standard modern Hindi.
    """
    if not text or not text.strip():
        return "", 1.0

    # 1. Apply pre-normalization dictionary for maximum alignment
    pre_cleaned_text = pre_normalize_text(text)
    
    # 2. If it's standard English, skip normalization
    if detected_dialect.lower() == "standard english":
        return text, 1.0

    # 3. Dynamic LLM Normalization Prompt
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    model_name = os.getenv("OCR_CORRECTION_MODEL", "phi3")

    prompt = f"""You are a specialized dialect normalization assistant serving the Rajasthan Cyber Cell.
Your task is to convert the following raw complaint text containing '{detected_dialect}' dialect terms into standard modern Hindi (in either Devanagari script or clean Romanized Transliteration, matching the input script style).

STRICT NORMALIZATION RULES:
1. Fix dialect spelling corruptions (e.g. 'mharo' -> 'mera', 'kat gayo' -> 'kat gaya', 'konya' -> 'nahi').
2. Keep the raw sentence flow and structure EXACTLY. DO NOT summarize, compress, or write a summary.
3. DO NOT translate to English yet. Preserve Hindi words and grammar style.
4. Strictly protect these critical fields:
{f"- Extracted Entities: {extracted_entities}" if extracted_entities else "None"}
5. Output ONLY the normalized standard Hindi text. Do not add explanations or markdown.

Raw Text:
\"\"\"
{pre_cleaned_text}
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
            normalized_result = response.json().get("response", "").strip()
            if normalized_result:
                return normalized_result, 0.90
    except Exception as exc:
        logger.error("Dialect LLM normalization failed: %s. Falling back to pre-normalized text.", exc)
        
    return pre_cleaned_text, 0.65
