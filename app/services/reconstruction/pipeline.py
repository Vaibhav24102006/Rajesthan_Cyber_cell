import logging
import os
import re
from time import perf_counter
from typing import Dict, Any, Tuple
import requests

from app.services.extractor import regex_extract
from app.services.reconstruction.language_detection import detect_regional_language
from app.services.reconstruction.dialect_normalization import pre_normalize_text
from app.services.reconstruction.translation import TRANSLATION_CACHE
from app.services.reconstruction.preservation_guardrails import (
    audit_entity_preservation,
    verify_and_restore_entities,
)
from app.services.reconstruction.benchmarking import calculate_benchmarks
from app.services.ocr_correction import is_ollama_reachable, get_active_correction_model

logger = logging.getLogger(__name__)

DOMAIN_PROMPTS = {
    "educational_content": "DOMAIN: Educational / Academic Notes. Terminology is likely historical, scientific, or academic (e.g., dynasty names like 'Mauryan', architecture terms like 'Sanchi stupas', equations, etc.). Reconstruct the language to be grammatically correct, clean, and highly readable academic notes. Correct OCR corruptions into proper educational and academic terms contextually.",
    "cybercrime_complaint": "DOMAIN: Cyber Crime / Online Fraud Complaint. Terminology likely includes Indian digital fraud keywords (Hinglish phrases like 'paise cut gaye', 'link open kiya', transaction details, 'Telegram scam', WhatsApp group, UPI, APK installation, etc.). Reconstruct this into a clear, professional, and readable cyber fraud report suitable for police investigation, detailing exactly how the incident occurred.",
    "banking_fraud": "DOMAIN: Banking Fraud / General Bank Complaint. The language relates to bank accounts, transfers, cheque issues, ATM disputes, banking branch complaints, and transactional discrepancies. Reconstruct this into a highly formal, clear banking complaint letter or report, correcting broken sentences.",
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

def offline_translate_hindi_hinglish_to_english(text: str) -> str:
    """
    Rule-based local translation fallback for common Hinglish/dialect cyber fraud expressions.
    Ensures Rajasthan complaints are readable even when Ollama is completely offline.
    """
    replacements = {
        "mharo bank khata se paise kat gaya": "money was deducted from my bank account",
        "mharo bank account se paise nikal gaye": "money was withdrawn from my bank account",
        "mera bank khata se paise kat gaya": "money was deducted from my bank account",
        "mera bank account se paise nikal gaye": "money was withdrawn from my bank account",
        "galti se click ho gaya": "clicked on it by mistake",
        "link open kiya": "opened the link",
        "upi pin share kiya": "shared the UPI PIN",
        "paise cut gaye": "money was deducted",
        "mharo paisa kat gayo": "my money was deducted",
        "telegram par contact kiya": "contacted via Telegram",
        "whatsapp par bol raha tha": "was speaking on WhatsApp",
        
        "bank khata": "bank account",
        "bank account": "bank account",
        "paise kat gaya": "money was deducted",
        "paise kat gaye": "money was deducted",
        "paise cut gaye": "money was deducted",
        "paise nikal gaye": "money was withdrawn",
        "khata se": "from account",
        "account se": "from account",
        "kat gayo": "was deducted",
        "kat gaya": "was deducted",
        "nikal gaya": "was withdrawn",
        "rupiya": "rupees",
        "rupye": "rupees",
        "dhokha": "fraud",
        "thagi": "fraud",
        "galti se": "by mistake",
        "click kiya": "clicked",
        "click ho gaya": "clicked",
        "open kiya": "opened",
        "bhej diya": "sent",
        "transfer kiya": "transferred",
    }
    
    text_lower = text.lower()
    cleaned_lower = text_lower.strip().rstrip(".,?!")
    
    # Check exact translation cache
    if cleaned_lower in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cleaned_lower]
        
    result = text
    # Run replacements in descending order of length to preserve compound phrases
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    for src, dest in sorted_replacements:
        pattern = re.compile(re.escape(src), re.IGNORECASE)
        result = pattern.sub(dest, result)
        
    return result

def build_consolidated_prompt(
    raw_text: str,
    detected_lang: str,
    detected_dialect: str,
    reconstruction_mode: str,
    document_domain: str,
    ocr_confidence: float,
    raw_entities: Dict[str, Any]
) -> str:
    domain_desc = DOMAIN_PROMPTS.get(document_domain, DOMAIN_PROMPTS["notes_documentation"])
    
    # Format mode instruction
    if reconstruction_mode == "structured_english_output":
        mode_desc = """RECONSTRUCTION MODE: Structured English Output.
- Convert any regional Rajasthani (Marwari, Mewari, Shekhawati) dialects and Hinglish expressions into modern standard standard Hindi first, then translate into formal, standard investigative English.
- Correct spelling mistakes, spacing issues, and handwriting OCR typos contextually.
- Clean up sentence flow while strictly retaining all facts and proper names."""
    elif reconstruction_mode == "normalized_hindi":
        mode_desc = """RECONSTRUCTION MODE: Normalized Hindi.
- Normalize Rajasthani dialect expressions or Hinglish terms into modern standard Hindi in Devanagari script (or matching script).
- Do NOT translate to English.
- Correct spelling mistakes, spacing issues, and handwriting OCR typos contextually."""
    elif reconstruction_mode == "preserve_original_language":
        mode_desc = """RECONSTRUCTION MODE: Preserve Original Language.
- Do NOT translate or normalize vocabulary. Maintain original sentence wording and dialects.
- Correct obvious spacing errors, punctuation, and casing errors ONLY. Do not guess noisy words."""
    else:
        mode_desc = MODE_PROMPTS.get(reconstruction_mode, MODE_PROMPTS["balanced_reconstruction"])

    conf_desc = get_confidence_instruction(ocr_confidence)

    # Format deterministic fields
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
        val = raw_entities.get(key)
        if val:
            deterministic_fields.append(f"- {label}: {val}")
    
    deterministic_fields_str = "\n".join(deterministic_fields) if deterministic_fields else "None identified in raw OCR."

    lang_instruction = f"""LANGUAGE & DIALECT CONTEXT:
- Detected Language: {detected_lang}
- Detected Dialect/Transliteration: {detected_dialect}
- Convert regional words (e.g. 'mharo' -> 'mera', 'kat gayo' -> 'kat gaya', 'konya' -> 'nahi') to standard Hindi forms before final reconstruction."""

    prompt = f"""You are a specialized multilingual cybercrime intelligence assistant for the Rajasthan Cyber Cell.
Your task is to take noisy handwritten complaint OCR text, normalize dialect differences, repair semantic handwriting errors, and format it beautifully.

---
{lang_instruction}

---
{domain_desc}

---
{mode_desc}

---
{conf_desc}

---
DETERMINISTIC FIELDS (MUST BE PRESERVED EXACTLY AND NEVER HALLUCINATED):
Below are the critical fields extracted from the raw document. You MUST preserve these exact strings/numbers in the final text. NEVER alter, drop, or guess them:
{deterministic_fields_str}

---
FORMATTING & LAYOUT RULES:
1. Reorganize the narrative chronologically into clean, formal, highly readable paragraphs.
2. Group key information like financial transaction details or phone numbers in a structured bold bullet layout.
3. Use headers like **Incident Details**, **Financial Details**, **Critical Evidence** where appropriate.
4. Output ONLY the reconstructed/formatted text. Do not add intro/outro comments, chatbot chat headers, or markdown code blocks like ```txt.

Raw Text:
\"\"\"
{raw_text}
\"\"\"
"""
    return prompt


def _clean_english_formatting(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return collapsed if collapsed else re.sub(r"\s+", " ", text).strip()


def _route_for_language(detected_lang: str, detected_dialect: str) -> str:
    lang = (detected_lang or "").lower()
    dialect = (detected_dialect or "").lower()
    if lang == "english" and dialect == "standard english":
        return "english_structuring_only"
    if lang in {"hindi"} or any(token in dialect for token in ("marwari", "mewari", "shekhawati", "braj")):
        return "regional_normalize_translate"
    if lang in {"hinglish", "mixed"}:
        return "selective_regional_translation"
    return "officer_review_required"


def _llm_enabled_for_language_route() -> bool:
    return os.getenv("LANGUAGE_ROUTING_ENABLE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}


def _fallback_translation_confidence(route: str, ocr_confidence: float) -> float:
    if route == "english_structuring_only":
        return 1.0
    if ocr_confidence < 0.60:
        return 0.35
    return 0.65


def reconstruct_ocr_text_staged(
    raw_text: str,
    ocr_confidence: float = 1.0,
    reconstruction_mode: str = "structured_english_output",
    document_domain: str = "auto"
) -> Tuple[str, bool, str, Dict[str, Any]]:
    """
    Language-aware OCR intelligence router.
    Detects the language/dialect first, then chooses the least-mutating safe path.
    """
    start_time = perf_counter()
    logger.info("Initializing language-aware OCR routing pipeline. Mode: %s", reconstruction_mode)

    # 1. Pass 1: Extracted entities from raw OCR
    raw_entities = regex_extract(raw_text)

    # 2. Pass 2: Language detection (instant, regex/dictionary-based)
    # Extract optional script-intelligence metadata if present
    script_info = raw_entities.pop("_script_detection", {}) if isinstance(raw_entities, dict) else {}
    visual_script = script_info.get("script") if isinstance(script_info, dict) else None
    script_confidence = script_info.get("confidence") if isinstance(script_info, dict) else None
    detected_lang, detected_dialect, lang_conf = detect_regional_language(
        raw_text, ocr_confidence,
        script_confidence=script_confidence,
        visual_script=visual_script,
    )
    logger.info("Detected Language: %s, Dialect: %s (Confidence: %.2f)", detected_lang, detected_dialect, lang_conf)

    routing_decision = _route_for_language(detected_lang, detected_dialect)
    warnings = []
    if ocr_confidence < 0.60:
        warnings.append("low_ocr_confidence_officer_review_required")
    if lang_conf < 0.70:
        warnings.append("low_language_confidence_officer_review_required")

    reconstructed_text = raw_text
    normalized_text = ""
    translated_text = ""
    fallback_used = False
    timeout_detected = False
    status = "language_route_accepted"
    reason = ""

    # Resolve active semantic domain
    active_domain = document_domain
    if active_domain == "auto":
        active_domain = detect_semantic_domain(raw_text)
    logger.info("Target Ingestion Domain: %s", active_domain)

    llm_enabled = _llm_enabled_for_language_route()
    ollama_active = bool(llm_enabled and is_ollama_reachable())

    if routing_decision == "english_structuring_only":
        status = "skipped_translation_english"
        reconstructed_text = _clean_english_formatting(raw_text)
        normalized_text = ""
        translated_text = ""
        logger.info("English route selected. Skipping multilingual reconstruction and translation.")
    elif ocr_confidence < 0.60:
        status = "low_confidence_deterministic_only"
        fallback_used = True
        normalized_text = pre_normalize_text(raw_text)
        translated_text = normalized_text if reconstruction_mode == "normalized_hindi" else offline_translate_hindi_hinglish_to_english(normalized_text)
        reconstructed_text = normalized_text if reconstruction_mode == "normalized_hindi" else translated_text
        logger.warning("Low OCR confidence route selected. Avoiding semantic LLM reconstruction.")
    elif not ollama_active:
        logger.info("Local deterministic language route selected. LLM enabled=%s active=%s.", llm_enabled, ollama_active)
        fallback_used = True
        status = "deterministic_language_fallback"

        normalized_text = pre_normalize_text(raw_text)
        if reconstruction_mode == "preserve_original_language":
            reconstructed_text = raw_text
            translated_text = ""
        elif reconstruction_mode == "normalized_hindi":
            reconstructed_text = normalized_text
            translated_text = ""
        else:
            translated_text = offline_translate_hindi_hinglish_to_english(normalized_text)
            reconstructed_text = translated_text
    else:
        # Consolidate dialect normalization, translation, repair, and formatting into ONE LLM CALL
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        model_name = get_active_correction_model()
        normalized_text = pre_normalize_text(raw_text)
        
        # Pre-apply entity preservation guardrails to the input to ensure safety
        pre_restored = verify_and_restore_entities(raw_text, raw_text, raw_entities)
        
        prompt = build_consolidated_prompt(
            raw_text=pre_restored,
            detected_lang=detected_lang,
            detected_dialect=detected_dialect,
            reconstruction_mode=reconstruction_mode,
            document_domain=active_domain,
            ocr_confidence=ocr_confidence,
            raw_entities=raw_entities
        )
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 1024,
            },
        }
        
        logger.info("Calling consolidated Ollama endpoint. Model: %s, Timeout: 5s", model_name)
        llm_start = perf_counter()
        try:
            # Enforce non-blocking 5-second socket timeout
            response = requests.post(ollama_url, json=payload, timeout=5.0)
            if response.ok:
                llm_out = response.json().get("response", "").strip()
                if llm_out:
                    reconstructed_text = llm_out
                    # Apply local translation/normalization cache sync for meta properties
                    if reconstruction_mode == "structured_english_output":
                        translated_text = llm_out
                    elif reconstruction_mode == "normalized_hindi":
                        normalized_text = llm_out
                        translated_text = ""
                else:
                    logger.warning("Ollama returned empty response. Falling back to local dictionary.")
                    fallback_used = True
                    status = "fallback_empty"
                    reconstructed_text = offline_translate_hindi_hinglish_to_english(normalized_text) if reconstruction_mode == "structured_english_output" else normalized_text
                    translated_text = reconstructed_text if reconstruction_mode == "structured_english_output" else ""
            else:
                logger.warning("Ollama call failed with status %s. Falling back to local dictionary.", response.status_code)
                fallback_used = True
                status = "fallback_status_error"
                reconstructed_text = offline_translate_hindi_hinglish_to_english(normalized_text) if reconstruction_mode == "structured_english_output" else normalized_text
                translated_text = reconstructed_text if reconstruction_mode == "structured_english_output" else ""
        except requests.exceptions.Timeout as exc:
            logger.error("Ollama connection timed out (5s limit reached). Gracefully falling back.")
            timeout_detected = True
            fallback_used = True
            status = "timeout_fallback_raw"
            reconstructed_text = offline_translate_hindi_hinglish_to_english(normalized_text) if reconstruction_mode == "structured_english_output" else normalized_text
            translated_text = reconstructed_text if reconstruction_mode == "structured_english_output" else ""
        except Exception as exc:
            logger.error("Ollama connection failed: %s. Gracefully falling back.", exc)
            fallback_used = True
            status = "error_fallback_raw"
            reconstructed_text = offline_translate_hindi_hinglish_to_english(normalized_text) if reconstruction_mode == "structured_english_output" else normalized_text
            translated_text = reconstructed_text if reconstruction_mode == "structured_english_output" else ""

    # Post-reconstruction Entity Preservation Verification (Guarantees deterministic fields are retained)
    pre_repair_audit = audit_entity_preservation(raw_text, reconstructed_text, raw_entities)
    reconstructed_text = verify_and_restore_entities(raw_text, reconstructed_text, raw_entities)
    entity_audit = audit_entity_preservation(raw_text, reconstructed_text, raw_entities)
    if pre_repair_audit.get("missing_entities"):
        entity_audit["status"] = "repaired"
    if entity_audit.get("preservation_rate", 1.0) < 1.0:
        warnings.append("entity_preservation_failed")

    # 4. Metrics & Benchmarks
    final_entities = regex_extract(reconstructed_text)
    benchmarks = calculate_benchmarks(raw_text, reconstructed_text, raw_entities, final_entities)
    benchmarks["entity_preservation_rate"] = entity_audit.get("preservation_rate", 1.0)
    
    total_pipeline_time_ms = round((perf_counter() - start_time) * 1000, 2)
    logger.info(
        "Language Routing Pipeline Completed in %.2f ms (Fallback: %s, Status: %s, Route: %s)",
        total_pipeline_time_ms,
        fallback_used,
        status,
        routing_decision,
    )

    translation_confidence = 0.88 if ollama_active and not fallback_used else _fallback_translation_confidence(routing_decision, ocr_confidence)
    semantic_confidence = 0.90 if routing_decision == "english_structuring_only" else (0.85 if ollama_active and not fallback_used else 0.60)
    normalized_regional_text = normalized_text if routing_decision != "english_structuring_only" else ""
    translation_output = translated_text if routing_decision != "english_structuring_only" else ""

    reconstruction_meta = {
        "reconstruction_model": get_active_correction_model() if ollama_active else "offline_local",
        "reconstruction_status": status,
        "reconstruction_source": "language_intelligence_router",
        "detected_language": detected_lang,
        "detected_dialect": detected_dialect,
        "dialect_detection": {
            "language": detected_lang,
            "dialect": detected_dialect,
            "confidence": lang_conf,
            "routing_decision": routing_decision,
        },
        "routing_decision": routing_decision,
        "normalized_dialect_text": normalized_text,
        "normalized_regional_text": normalized_regional_text,
        "translated_text": translated_text,
        "translation_output": translation_output,
        "reconstructed_text": reconstructed_text,
        "final_readable_text": reconstructed_text,
        "structured_english_complaint": reconstructed_text,
        "entity_preservation_audit": entity_audit,
        "language_route_confidence": lang_conf,
        "language_warnings": warnings,
        "stage_confidence": {
            "raw_ocr": ocr_confidence,
            "language_detection": lang_conf,
            "token_normalization": 1.0 if routing_decision == "english_structuring_only" else (0.90 if ollama_active and not fallback_used else 0.65),
            "translation": translation_confidence,
            "semantic_reconstruction": semantic_confidence,
            "entity_preservation": entity_audit.get("preservation_rate", 1.0),
            "formatting_confidence": 0.95
        },
        "benchmarks": benchmarks,
        "fallback_used": fallback_used,
        "timeout_detected": timeout_detected,
        "guardrail_status": entity_audit.get("status", "not_checked"),
        "reconstruction_confidence_score": 0.95 if routing_decision == "english_structuring_only" else (0.85 if not fallback_used else 0.60),
        "reconstruction_runtime_ms": total_pipeline_time_ms,
        "reconstruction_mode": reconstruction_mode,
        "detected_domain": active_domain,
    }

    return reconstructed_text, False, reason, reconstruction_meta
