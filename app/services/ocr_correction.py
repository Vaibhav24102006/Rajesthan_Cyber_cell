import logging
import os
from time import perf_counter
from typing import Any, Dict, Tuple

import requests

from app.services.ocr_guardrails import validate_ocr_correction
from app.services.extractor import regex_extract

logger = logging.getLogger(__name__)

OCR_CORRECTION_MODEL = "phi3"
OCR_CORRECTION_TIMEOUT_SECONDS = float(os.getenv("OCR_CORRECTION_TIMEOUT_SECONDS", "25"))


def get_active_correction_model() -> str:
    return os.getenv("OCR_CORRECTION_MODEL", os.getenv("OLLAMA_OCR_MODEL", OCR_CORRECTION_MODEL))


def is_ollama_reachable() -> bool:
    health_url = os.getenv("OLLAMA_HEALTH_URL", "http://localhost:11434/api/tags")
    try:
        resp = requests.get(health_url, timeout=5)
        return resp.ok
    except Exception:
        return False


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


def correct_ocr_text(
    raw_text: str,
    ocr_confidence: float = 1.0,
    reconstruction_mode: str = "balanced_reconstruction",
    document_domain: str = "notes_documentation"
) -> Tuple[str, bool, str, Dict[str, Any]]:
    """
    Passes the raw OCR text through a dynamic, context-aware local LLM via Ollama.
    Supports chunking for documents over 2000 characters.
    """
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    model_name = get_active_correction_model()

    extracted_entities = regex_extract(raw_text)

    start = perf_counter()
    correction_meta: Dict[str, Any] = {
        "correction_model": model_name,
        "reconstruction_model": model_name,
        "correction_status": "not_started",
        "timeout_detected": False,
        "fallback_used": False,
        "guardrail_status": "not_checked",
        "correction_runtime_ms": 0.0,
        "detected_domain": document_domain,
        "reconstruction_mode": reconstruction_mode,
        "reconstruction_confidence_score": 0.85 if ocr_confidence >= 0.85 else (0.50 if ocr_confidence < 0.70 else 0.70),
    }

    def call_ollama(payload_prompt: str) -> str:
        payload = {
            "model": model_name,
            "prompt": payload_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 1024,
            },
        }
        response = requests.post(ollama_url, json=payload, timeout=OCR_CORRECTION_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
        return body.get("response", "").strip()

    try:
        if len(raw_text) > 2000:
            logger.info("Local OCR correction text length (%s) exceeds 2000 characters, running chunk-based reconstruction.", len(raw_text))
            paragraphs = raw_text.split("\n\n")
            chunks = []
            current_chunk = []
            current_len = 0
            for p in paragraphs:
                p_len = len(p)
                if current_len + p_len > 1500 and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [p]
                    current_len = p_len
                else:
                    current_chunk.append(p)
                    current_len += p_len + 2
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            reconstructed_chunks = []
            for idx, chunk in enumerate(chunks):
                prompt = _build_advanced_prompt(
                    chunk,
                    ocr_confidence,
                    reconstruction_mode,
                    document_domain,
                    extracted_entities
                )
                if len(chunks) > 1:
                    prompt = f"NOTE: This is segment {idx + 1} of {len(chunks)} from a larger document.\n\n" + prompt
                
                chunk_text = call_ollama(prompt)
                reconstructed_chunks.append(chunk_text)

            corrected_text = "\n\n".join(reconstructed_chunks)
        else:
            prompt = _build_advanced_prompt(
                raw_text,
                ocr_confidence,
                reconstruction_mode,
                document_domain,
                extracted_entities
            )
            corrected_text = call_ollama(prompt)

        correction_meta["correction_runtime_ms"] = round((perf_counter() - start) * 1000, 2)

        if not corrected_text:
            correction_meta["correction_status"] = "fallback_raw_empty_response"
            correction_meta["fallback_used"] = True
            return raw_text, True, "Ollama returned empty response", correction_meta

        is_valid, reason = validate_ocr_correction(raw_text, corrected_text)
        if not is_valid:
            logger.warning("OCR Correction rejected: %s", reason)
            correction_meta["correction_status"] = "rejected_by_guardrails"
            correction_meta["guardrail_status"] = "rejected"
            correction_meta["fallback_used"] = True
            return raw_text, True, reason, correction_meta

        correction_meta["correction_status"] = "accepted"
        correction_meta["guardrail_status"] = "accepted"
        return corrected_text, False, "", correction_meta

    except requests.exceptions.ReadTimeout as exc:
        correction_meta["correction_runtime_ms"] = round((perf_counter() - start) * 1000, 2)
        correction_meta["correction_status"] = "timeout_fallback_raw"
        correction_meta["timeout_detected"] = True
        correction_meta["fallback_used"] = True
        logger.error("OCR Correction timeout: %s", exc)
        return raw_text, True, f"LLM timeout: {str(exc)}", correction_meta
    except Exception as exc:
        correction_meta["correction_runtime_ms"] = round((perf_counter() - start) * 1000, 2)
        correction_meta["correction_status"] = "error_fallback_raw"
        correction_meta["fallback_used"] = True
        logger.error("OCR Correction failed: %s", exc)
        return raw_text, True, f"LLM error: {str(exc)}", correction_meta
