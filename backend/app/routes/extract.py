import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.complaint import CaseResponse, Complaint, ExtractRequest, ExtractResponse
from app.services.ai_service import enrich_with_ai
from app.services.extractor import regex_extract


logger = logging.getLogger(__name__)
router = APIRouter()


def _merge_data(regex_data: Dict, ai_data: Dict) -> Dict:
    merged = {
        "victim_name": regex_data.get("victim_name") or ai_data.get("victim_name"),
        "phone": regex_data.get("phone") or ai_data.get("phone"),
        "location": regex_data.get("location") or ai_data.get("location"),
        "amount": regex_data.get("amount") or ai_data.get("amount"),
        "transaction_id": regex_data.get("transaction_id") or ai_data.get("transaction_id"),
        "account_number": regex_data.get("account_number") or ai_data.get("account_number"),
        "bank": regex_data.get("bank") or ai_data.get("bank"),
        "crime_type": ai_data.get("crime_type", "Other"),
        "severity": ai_data.get("severity", 4),
        "summary": ai_data.get("summary")
        or "Complaint received. Further verification needed by investigation officer.",
    }
    return merged


@router.post("/extract", response_model=ExtractResponse)
def extract_complaint(payload: ExtractRequest, db: Session = Depends(get_db)):
    try:
        logger.info("Processing complaint extraction request")
        regex_data = regex_extract(payload.text)
        ai_data = enrich_with_ai(payload.text, regex_data)
        merged_data = _merge_data(regex_data, ai_data)

        complaint = Complaint(**merged_data, raw_text=payload.text)
        db.add(complaint)
        db.commit()
        logger.info("Complaint saved with id=%s", complaint.id)

        return ExtractResponse(**merged_data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process complaint") from exc


@router.get("/cases", response_model=list[CaseResponse])
def list_cases(db: Session = Depends(get_db)):
    try:
        cases = db.query(Complaint).order_by(Complaint.created_at.desc()).all()
        return cases
    except Exception as exc:
        logger.exception("Failed to fetch cases: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch cases") from exc
