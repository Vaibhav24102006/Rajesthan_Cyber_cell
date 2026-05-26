import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

from app.services.ocr_correction import correct_ocr_text
from app.services.ocr_guardrails import validate_ocr_correction
from app.services.extractor import regex_extract

logger = logging.getLogger(__name__)

INDUS_API_URL = os.getenv("INDUS_API_URL", "").strip()
INDUS_API_KEY = os.getenv("INDUS_API_KEY", "").strip()
INDUS_MODEL = os.getenv("INDUS_MODEL", "handwriting-normalization").strip()
INDUS_TIMEOUT_SECONDS = float(os.getenv("INDUS_TIMEOUT_SECONDS", "25"))
INDUS_HEALTH_URL = os.getenv("INDUS_HEALTH_URL", INDUS_API_URL).strip()


def is_indus_reachable() -> bool:
    if not INDUS_HEALTH_URL:
        return False
    try:
        resp = requests.get(INDUS_HEALTH_URL, timeout=5)
        return resp.ok
    except Exception:
        return False


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_reconstruction_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_indus_response(body: Any) -> Tuple[str, float]:
    if body is None:
        return "", 0.0

    if isinstance(body, str):
        return _normalize_reconstruction_text(body), 0.0

    if isinstance(body, dict):
        confidence = 0.0
        for score_key in ["confidence", "score", "probability", "certainty"]:
            if score_key in body:
                confidence = _safe_float(body.get(score_key))
                break

        candidates = []
        if "reconstructed_text" in body:
            candidates.append(body["reconstructed_text"])
        if "output" in body:
            candidates.append(body["output"])
        if "response" in body:
            candidates.append(body["response"])
        if "text" in body:
            candidates.append(body["text"])
        if "result" in body:
            candidates.append(body["result"])

        predictions = body.get("predictions")
        if isinstance(predictions, list) and predictions:
            first_prediction = predictions[0]
            if isinstance(first_prediction, dict):
                for key in ["output", "text", "content", "response", "reconstructed_text", "result"]:
                    if key in first_prediction:
                        candidates.append(first_prediction[key])
                if "data" in first_prediction and isinstance(first_prediction["data"], str):
                    candidates.append(first_prediction["data"])
            elif isinstance(first_prediction, str):
                candidates.append(first_prediction)

        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                for key in ["text", "message", "output_text", "content"]:
                    if key in first_choice:
                        candidates.append(first_choice[key])
            elif isinstance(first_choice, str):
                candidates.append(first_choice)

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return _normalize_reconstruction_text(candidate), min(max(confidence, 0.0), 1.0)

    return "", 0.0


DOMAIN_PROMPTS = {
    "educational_content": "DOMAIN: Educational / Academic Notes. Terminology is likely historical, scientific, or academic (e.g., dynasty names like 'Mauryan', architecture terms like 'Sanchi stupas', equations, etc.). Reconstruct the language to be grammatically correct, clean, and highly readable academic notes. Correct OCR corruptions into proper educational and academic terms contextually.",
    "cybercrime_complaint": "DOMAIN: Cyber Crime / Online Fraud Complaint. Terminology likely includes Indian digital fraud keywords (Hinglish phrases like 'paise cut gaye', 'link open kiya', transaction details, 'Telegram scam', WhatsApp group, UPI, APK installation, etc.). Reconstruct this into a clear, professional, and readable cyber fraud report suitable for police investigation, detailing exactly how the incident occurred.",
    "banking_fraud": "DOMAIN: Banking Fraud / General Bank Complaint. The language relates to bank accounts, transfers, cheque issues, ATM disputes, branch complaints, and transactional discrepancies. Reconstruct this into a highly formal, clear banking complaint letter or report, correcting broken sentences.",
    "legal_complaint": "DOMAIN: Legal / FIR-style Police Complaint. The tone should be official, formal, and police-friendly. Reconstruct the noisy OCR text into a clear, chronological, and highly readable formal police complaint statement. Correct spelling, grammar, and sentence structure, but strictly keep all allegations and events exactly as described.",
    "mixed_hindi_english": "DOMAIN: Mixed Hindi-English (Hinglish) Complaint. The text uses common Indian expressions transliterated into Latin characters (e.g., 'mere bank se paise nikal gaye', 'galti se click ho gaya'). Reconstruct this into a highly readable, clear, grammatically cohesive, and investigation-ready Hindi-English or standard Hinglish format, resolving noisy transliterated tokens to their intended words.",
    "notes_documentation": "DOMAIN: General Notes / Personal Documentation. Reconstruct this text into highly readable, clean, grammatically correct common language, resolving OCR noise and spelling mistakes contextually."
}

MODE_PROMPTS = {
    "strict_preservation": """RECONSTRUCTION MODE: Strict Preservation.
- Highly conservative. Fix obvious spelling typos, spacing, and casing errors ONLY.
- If a word is noisy, heavily corrupted, or highly uncertain, DO NOT try to guess or reconstruct it. Wrap it in [brackets] (e.g., "[hgjksld]") to indicate uncertainty.
- Prioritize literal transcription accuracy and factual preservation over readability and flow.
- Maintain original sentence structure exactly.""",

    "balanced_reconstruction": """RECONSTRUCTION MODE: Balanced Reconstruction.
- Balanced repair and flow.
- Fix spacing, grammar, transliteration errors, and spelling mistakes.
- For noisy or corrupted words, use context clues and domain knowledge to infer the likely intended words (e.g., "Mauyn archttue" -> "Mauryan architecture").
- Do not rewrite entire sentences. If a whole sentence is complete gibberish, keep it in [brackets], but repair individual corrupted words.
- Maintain a natural flow while staying close to the source text.""",

    "aggressive_semantic_repair": """RECONSTRUCTION MODE: Aggressive Semantic Repair.
- Aggressively repair noisy handwriting OCR, malformed grammar, and broken Hinglish.
- Your primary goal is to generate a highly READABLE, COMMON LANGUAGE output that is clear and immediately usable for investigations.
- Do not preserve corrupted literal tokens or use bracketed placeholders [like this] unless a phrase is completely undecipherable.
- Infer likely intended words based on context and domain knowledge to produce highly polished, professional text. Reconstruct full sentences to flow naturally and sound coherent."""
}


def get_confidence_instruction(ocr_confidence: float) -> str:
    if ocr_confidence < 0.70:
        return f"""CONFIDENCE AWARENESS (OCR Confidence: {ocr_confidence:.2f} - LOW):
- The OCR confidence is very low, meaning the raw text is highly noisy and contains heavily corrupted tokens.
- You MUST rely heavily on context clues, semantic understanding, and domain knowledge to reconstruct clean, readable, coherent language.
- Minimize literal garbage tokens; aggressively infer the intended words and reconstruct readable common language."""
    elif ocr_confidence >= 0.85:
        return f"""CONFIDENCE AWARENESS (OCR Confidence: {ocr_confidence:.2f} - HIGH):
- The OCR confidence is high, meaning the raw text is relatively clean.
- Focus on preserving the literal structure and wording closely, fixing only minor spacing, spelling, and obvious typos. Do not over-reconstruct."""
    else:
        return f"""CONFIDENCE AWARENESS (OCR Confidence: {ocr_confidence:.2f} - BALANCED):
- The OCR confidence is moderate.
- Apply balanced correction. Keep valid words and structure, but reconstruct noisy tokens using semantic context clues."""


def detect_semantic_domain(text: str) -> str:
    """
    Analyzes the text using high-confidence keyword heuristics to detect the semantic domain.
    """
    text_lower = text.lower()
    
    # 1. Cybercrime Complaint
    cybercrime_keywords = [
        "transaction", "txn", "upi", "hack", "facebook", "whatsapp", "otp", "cyber", "fraud", "scam", 
        "telegram", "website", "link", "phishing", "cryptocurrency", "crypto", "wallet", "instagram",
        "online fraud", "cyber cell", "deducted", "lost money", "jamtara", "apk", "extracted"
    ]
    cyber_score = sum(1 for kw in cybercrime_keywords if kw in text_lower)
    
    # 2. Banking Fraud / Complaint
    banking_keywords = [
        "bank", "account", "a/c", "ifsc", "passbook", "branch", "cheque", "atm", "debit", "credit", 
        "loan", "card", "deposit", "transfer", "hdfc", "sbi", "icici", "pnb", "bob", "axis bank"
    ]
    banking_score = sum(1 for kw in banking_keywords if kw in text_lower)
    
    # 3. Legal / FIR / Police Complaint
    legal_keywords = [
        "fir", "police", "complaint", "station", "si", "inspector", "accused", "ipc", "crpc", 
        "section", "theft", "assault", "formal", "accuse", "arrest", "court", "complaining",
        "than", "thana", "shriman", "sho", "prarthna", "patra", "shapath"
    ]
    legal_score = sum(1 for kw in legal_keywords if kw in text_lower)
    
    # 4. Educational Content / Notes
    educational_keywords = [
        "architecture", "mauryan", "notes", "dynasty", "history", "science", "math", "chapter", 
        "lecture", "student", "class", "study", "sanchi", "stupa", "ashoka", "emperor", "king",
        "civilization", "harappan", "gupta", "buddhism", "jainism"
    ]
    edu_score = sum(1 for kw in educational_keywords if kw in text_lower)
    
    scores = {
        "cybercrime_complaint": cyber_score,
        "banking_fraud": banking_score,
        "legal_complaint": legal_score,
        "educational_content": edu_score,
    }
    
    max_domain = max(scores, key=scores.get)
    if scores[max_domain] > 0:
        return max_domain
        
    return "notes_documentation"


def _build_advanced_prompt(
    raw_text: str,
    ocr_confidence: float,
    reconstruction_mode: str,
    document_domain: str,
    extracted_entities: Dict[str, Any]
) -> str:
    domain_desc = DOMAIN_PROMPTS.get(document_domain, DOMAIN_PROMPTS["notes_documentation"])
    mode_desc = MODE_PROMPTS.get(reconstruction_mode, MODE_PROMPTS["balanced_reconstruction"])
    conf_desc = get_confidence_instruction(ocr_confidence)
    
    deterministic_fields = []
    field_labels = {
        "phone": "Phone Numbers",
        "transaction_id": "Transaction IDs",
        "account_number": "Bank Accounts",
        "upi_id": "UPI IDs",
        "ifsc": "IFSC Codes",
        "date": "Dates",
        "url": "URLs",
        "amount": "Financial Amounts"
    }
    
    for key, label in field_labels.items():
        val = extracted_entities.get(key)
        if val:
            deterministic_fields.append(f"- {label}: {val}")
            
    if deterministic_fields:
        deterministic_fields_str = "\n".join(deterministic_fields)
    else:
        deterministic_fields_str = "None identified in raw OCR."

    prompt = f"""You are a forensic handwriting normalization engine serving Rajasthan Cyber Cell.
Input is noisy OCR text extracted from a handwritten document.
Your job is to reconstruct the text as clean, highly readable language optimized for investigation usability and human readability, prioritizing readable common language over literal OCR token preservation.

---
{domain_desc}

---
{mode_desc}

---
{conf_desc}

---
DETERMINISTIC FIELDS (MUST BE PRESERVED EXACTLY):
Below are the critical fields extracted from the raw document. You MUST preserve these exact strings/numbers in the reconstructed text if they appear. NEVER alter, drop, or guess them:
{deterministic_fields_str}

STRICT GENERAL RULES:
1. Never summarize or shorten the complaint; preserve all details.
2. Never invent, hallucinate, or add new entities, bank names, or transaction values.
3. Repair broken words, normalize spacing, and normalize Hinglish/mixed-language text.
4. Output ONLY the reconstructed text. Do not add markdown blocks like ```txt or commentary.

Raw OCR Input:
\"\"\"
{raw_text}
\"\"\"
"""
    return prompt


def _build_indus_payload(raw_text: str, prompt: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "input": raw_text,
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "instructions": prompt,
    }
    if INDUS_MODEL:
        payload["model"] = INDUS_MODEL
    return payload


from app.services.reconstruction import reconstruct_ocr_text_staged

def reconstruct_ocr_text(
    raw_text: str,
    ocr_confidence: float = 1.0,
    reconstruction_mode: str = "balanced_reconstruction",
    document_domain: str = "auto",
) -> Tuple[str, bool, str, Dict[str, Any]]:
    """
    Delegates/proxies the existing reconstruct_ocr_text to the 6-stage multilingual pipeline,
    guaranteeing complete backward-compatibility and regional-language capabilities.
    """
    logger.info("Proxying reconstruct_ocr_text request to reconstruct_ocr_text_staged")
    return reconstruct_ocr_text_staged(
        raw_text=raw_text,
        ocr_confidence=ocr_confidence,
        reconstruction_mode=reconstruction_mode,
        document_domain=document_domain
    )



def _fallback_to_local(
    raw_text: str,
    reason: str,
    ocr_confidence: float = 1.0,
    reconstruction_mode: str = "balanced_reconstruction",
    detected_domain: str = "notes_documentation"
) -> Tuple[str, bool, str, Dict[str, Any]]:
    corrected_text, was_rejected, rejection_reason, correction_meta = correct_ocr_text(
        raw_text,
        ocr_confidence=ocr_confidence,
        reconstruction_mode=reconstruction_mode,
        document_domain=detected_domain
    )
    correction_meta["reconstruction_source"] = "local_correction_fallback"
    correction_meta["reconstruction_status"] = "fallback_local"
    correction_meta["reconstruction_confidence_score"] = correction_meta.get("reconstruction_confidence_score", 0.0)
    correction_meta["reconstruction_runtime_ms"] = correction_meta.get("correction_runtime_ms", 0.0)
    correction_meta["timeout_detected"] = correction_meta.get("timeout_detected", False)
    correction_meta["fallback_used"] = True
    correction_meta["guardrail_status"] = correction_meta.get("guardrail_status", "not_checked")
    correction_meta["rejection_reason"] = reason
    correction_meta["detected_domain"] = detected_domain
    correction_meta["reconstruction_mode"] = reconstruction_mode
    return corrected_text, was_rejected, rejection_reason or reason, correction_meta
