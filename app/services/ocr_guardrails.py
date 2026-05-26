import logging
from typing import Dict, Any, Tuple

from app.services.extractor import regex_extract

logger = logging.getLogger(__name__)

def validate_ocr_correction(raw_text: str, corrected_text: str) -> Tuple[bool, str]:
    """
    Compares the deterministic entities extracted from the raw OCR vs the AI-corrected OCR.
    Returns (is_valid, rejection_reason).
    """
    raw_entities = regex_extract(raw_text)
    corrected_entities = regex_extract(corrected_text)

    # 1. Numeric Preservation
    # If a valid numeric entity (phone, account, txn, amount) existed in raw, 
    # it must exist in corrected and not be completely rewritten.
    # Note: If it didn't exist in raw (due to OCR noise like 'BG' instead of '36'),
    # but exists in corrected, that's considered a successful fix, so we allow it.
    
    strict_preservation_keys = [
        "phone",
        "account_number",
        "transaction_id",
        "amount"
    ]

    for key in strict_preservation_keys:
        raw_val = raw_entities.get(key)
        corr_val = corrected_entities.get(key)
        
        # If raw had a valid extraction, the corrected must preserve it exactly.
        if raw_val and corr_val:
            if raw_val != corr_val:
                # The AI changed a successfully validated number. This is a hallucination.
                return False, f"Modified validated {key}: {raw_val} -> {corr_val}"
        
        # If raw had a valid extraction but corrected lost it entirely
        if raw_val and not corr_val:
            return False, f"Lost validated {key}: {raw_val}"

    # 2. Bank Preservation
    raw_bank = raw_entities.get("bank")
    corr_bank = corrected_entities.get("bank")
    if raw_bank and corr_bank:
        if raw_bank != corr_bank:
            return False, f"Modified validated bank: {raw_bank} -> {corr_bank}"
    if raw_bank and not corr_bank:
        return False, f"Lost validated bank: {raw_bank}"

    return True, ""
