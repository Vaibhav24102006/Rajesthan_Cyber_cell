import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

DIALECT_KEYWORDS = {
    "marwari": [
        "mharo", "mhara", "tharo", "thara", "mhane", "thane", "kathe", "jaye", "hove", "kun", "koni",
        "chhoro", "chhori", "rupiya", "rupye", "kat gayo", "aayo", "gayo", "kar gayo", "karlo", "konya",
        "mharo", "mhara", "mhane",
    ],
    "mewari": [
        "mhanu", "thanu", "vuno", "kune", "nhin", "nhvya", "karvyo", "avya", "bhado", "katgyo", "vuna",
    ],
    "shekhawati": [
        "kathin", "jathin", "koini", "lugai", "laado", "baatan", "rupyaji", "katai", "karis", "karego",
    ],
    "braj": [
        "mohe", "tohe", "mero", "tero", "kahe", "kachu", "nahin", "bhayo", "gayo", "ayo", "paiso",
        "khato", "thagyo", "batayo", "karayo",
    ],
    "hinglish": [
        "paise", "khata", "kat", "gaya", "gaye", "account", "click", "kiya", "scam", "link", "aya", "galti",
        "kiya", "dhokha", "hacker", "telegram", "whatsapp", "call", "group", "bol", "raha", "tha",
    ],
}

HINDI_STOPWORDS = {
    "mera", "mere", "mujhe", "maine", "paise", "rupaye", "khata", "bank", "se", "kat", "gaye",
    "nikal", "gaya", "dhokha", "shikayat", "kripya", "karyavahi", "police", "thana", "nivedan",
    "mahtvapurn", "mahoday", "shriman", "jankari", "kripya", "praye", "vishey",
}

DEVANAGARI_STRUCTURAL_PATTERNS = [
    re.compile(r"[क-ह]"),       # Consonants
    re.compile(r"[ा-ौ]"),       # Matras (vowel signs)
    re.compile(r"[\u0900-\u0904]"),  # Chandrabindu, anusvara, visarga, etc.
    re.compile(r"[ऀ-ऋ]"),       # Vowels
    re.compile(r"[ए-औ]"),       # Independent vowels
]

ENGLISH_WORDS = {
    "the", "and", "account", "transaction", "credited", "debited", "complaint", "police", "cyber",
    "fraud", "message", "received", "mobile", "number", "bank", "application", "insurance",
    "deducted", "investigate", "amount", "transfer", "of", "on", "from", "in", "as", "was",
    "impact", "different", "temple", "architecture", "religion", "religions", "scenes",
    "history", "notes", "philosophy", "philosophies", "please", "help", "thank", "dear",
    "sir", "madam", "request", "information", "subject", "reference", "regarding",
}

MARWARI_KEYWORDS_EXTENDED = {
    "mharo", "mhara", "mhane", "tharo", "thara", "thane", "kathe", "jaye", "hove", "kun",
    "koni", "konya", "rupiya", "rupye", "kat gayo", "aayo", "gayo", "kar gayo", "karlo",
    "chhoro", "chhori", "lugai", "laado", "baatan", "paani", "paiso", "katai", "karego",
    "karis", "bhado", "avya", "karvyo", "katgyo", "vuno", "vuna", "kune", "nhin", "nhvya",
    "mhanu", "thanu", "kathin", "jathin", "koini", "mohe", "tohe", "mero", "tero", "kahe",
    "kachu", "nahin", "bhayo", "paiso", "khato", "thagyo", "batayo", "karayo",
}


def _script_counts(text: str) -> Dict[str, int]:
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    return {"devanagari": devanagari, "latin": latin, "digits": digits}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u0900-\u097FA-Za-z]+", text.lower())


def _devanagari_structure_score(text: str) -> float:
    if not text:
        return 0.0
    dev_chars = len(re.findall(r"[\u0900-\u097F]", text))
    if dev_chars == 0:
        return 0.0
    consonant_count = len(re.findall(r"[क-ह]", text))
    matra_count = len(re.findall(r"[ा-ौ]", text))
    vowel_count = len(re.findall(r"[ऀ-ऋ]", text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return 0.0
    structural_ratio = (consonant_count + matra_count + vowel_count) / max(1, dev_chars)
    return min(1.0, structural_ratio * 1.2)


def _regional_token_prior(text: str) -> float:
    text_lower = text.lower()
    marwari_hits = sum(1 for kw in MARWARI_KEYWORDS_EXTENDED if kw in text_lower)
    if marwari_hits >= 3:
        return 0.95
    if marwari_hits >= 1:
        return 0.70
    return 0.0


def detect_regional_language(
    text: str,
    ocr_confidence: float = 1.0,
    script_confidence: float = None,
    visual_script: str = None,
) -> Tuple[str, str, float]:
    if not text or not text.strip():
        if script_confidence and script_confidence > 0.3 and visual_script == "devanagari":
            return "Hindi", "Standard Hindi (visual detection only)", script_confidence * 0.6
        return "unknown", "unknown", 0.0

    text_lower = text.lower()
    tokens = _tokenize(text)
    token_set = set(tokens)
    scripts = _script_counts(text)
    alpha_total = max(1, scripts["devanagari"] + scripts["latin"])
    devanagari_ratio = scripts["devanagari"] / alpha_total
    latin_ratio = scripts["latin"] / alpha_total

    has_devanagari = scripts["devanagari"] > 0

    # ─── SCRIPT CONFIDENCE (from actual OCR output) ───
    dev_struct_score = _devanagari_structure_score(text)
    combined_dev_confidence = devanagari_ratio * 0.4 + dev_struct_score * 0.4 + (0.2 if has_devanagari else 0.0)

    # ─── VISUAL SCRIPT PRIOR ───
    # If visual detection says Devanagari but OCR produced garbage, override
    visual_override = False
    if visual_script == "devanagari" and script_confidence and script_confidence > 0.40:
        if combined_dev_confidence < 0.20:
            logger.info(
                "Language detection override: visual script=devanagari (conf=%.2f) but OCR produced "
                "low Devanagari ratio (%.4f). Using visual prior to identify Hindi structure.",
                script_confidence, devanagari_ratio
            )
            visual_override = True

    # Calculate keyword match frequencies
    scores = {
        "marwari": sum(1 for kw in DIALECT_KEYWORDS["marwari"] if kw in text_lower),
        "mewari": sum(1 for kw in DIALECT_KEYWORDS["mewari"] if kw in text_lower),
        "shekhawati": sum(1 for kw in DIALECT_KEYWORDS["shekhawati"] if kw in text_lower),
        "braj": sum(1 for kw in DIALECT_KEYWORDS["braj"] if kw in text_lower),
        "hinglish": sum(1 for kw in DIALECT_KEYWORDS["hinglish"] if kw in text_lower),
    }

    english_score = sum(1 for ew in ENGLISH_WORDS if ew in token_set)
    hindi_roman_score = sum(1 for hw in HINDI_STOPWORDS if hw in token_set)

    regional_score = _regional_token_prior(text)

    regional_scores = {key: val for key, val in scores.items() if key != "hinglish"}
    max_regional_dialect = max(regional_scores, key=regional_scores.get) if regional_scores else "hinglish"
    max_regional_score = regional_scores.get(max_regional_dialect, 0) if regional_scores else 0
    if max_regional_score >= 2:
        max_dialect = max_regional_dialect
        max_score = max_regional_score
    else:
        max_dialect = max(scores, key=scores.get)
        max_score = scores[max_dialect]

    # ─── LANGUAGE CLASSIFICATION ───
    # CASE 1: Visual override — we know it's Devanagari but OCR collapsed it
    if visual_override:
        if regional_score > 0.50:
            dialect = "Marwari" if scores.get("marwari", 0) >= scores.get("mewari", 0) else "Mewari"
            return "Hindi", dialect, max(0.6, script_confidence * 0.8)
        if hindi_roman_score >= 2:
            return "Hinglish", "Hinglish", max(0.55, script_confidence * 0.7)
        return "Hindi", "Hindi (OCR collapsed, visual override)", max(0.50, script_confidence * 0.65)

    # CASE 2: Clear English (high Latin, high English vocabulary, no Hindi)
    if latin_ratio > 0.80 and english_score >= 5 and hindi_roman_score <= 1 and max_regional_score == 0:
        return "English", "Standard English", 0.96

    if latin_ratio > 0.80 and not has_devanagari and english_score >= 3 and hindi_roman_score <= 1 and max_regional_score == 0:
        return "English", "Standard English", 0.88

    # CASE 3: Has Devanagari characters
    if has_devanagari:
        if devanagari_ratio >= 0.65 and max_score == 0:
            if regional_score > 0.50:
                return "Hindi", "Marwari-influenced Hindi", max(0.85, regional_score)
            return "Hindi", "Standard Hindi", 0.92

        if devanagari_ratio >= 0.40:
            if max_score > 0:
                base = min(0.80 + max_score * 0.05, 0.98)
                if regional_score > 0.50:
                    return "Hindi", "Marwari", max(base, regional_score)
                return "Hindi", max_dialect.capitalize(), base
            return "Hindi", "Standard Hindi", 0.85

        if latin_ratio > 0.20:
            if max_score > 0:
                return "Mixed", f"Hindi-{max_dialect.capitalize()} Mixed", 0.78
            return "Mixed", "Hindi-English Mixed", 0.78

    # CASE 4: No Devanagari, check dialect keywords
    if max_score > 0:
        confidence = min(0.50 + (max_score * 0.15), 0.98)
        if max_dialect == "hinglish":
            return "Hinglish", "Hinglish", confidence
        return "Hinglish", f"Romanized {max_dialect.capitalize()}", confidence

    # CASE 5: Latin text with English vocabulary
    if english_score >= 3 and hindi_roman_score == 0:
        return "English", "Standard English", 0.95

    if hindi_roman_score >= 2:
        base_conf = min(0.70 + hindi_roman_score * 0.05, 0.90)
        if regional_score > 0.50:
            return "Hinglish", "Marwari-influenced Hinglish", max(base_conf, regional_score * 0.8)
        return "Hinglish", "Hinglish", base_conf

    # CASE 6: Fallback with regional prior
    if regional_score > 0.30:
        return "Hindi", "Hindi (regional priors)", regional_score * 0.7

    if len(text.strip()) > 10:
        return "Mixed", "Mixed Dialect", 0.65

    return "unknown", "unknown", 0.30
