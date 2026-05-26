import hashlib
import logging
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.complaint import CaseResponse, Complaint, ExtractRequest, ExtractResponse
from app.services.ai_service import enrich_with_ai
from app.services.extractor import deterministic_extract_with_confidence
from app.services.ner_service import ner_extract
from app.services.cyber_entities import parse_cyber_entities
from app.services.crime_taxonomy import classify_by_taxonomy
from app.services.ocr_service import extract_text_from_document, get_documents_dir, OCRRuntimeError
from app.models.audit_log import AuditLog


logger = logging.getLogger(__name__)
router = APIRouter()


def _merge_data(regex_data: Dict, ai_data: Dict) -> Dict:
    # Regex is deterministic (high confidence). AI is semantic (lower confidence).
    # We compute confidence_score from which fields were populated.
    key_fields = [
        "victim_name",
        "phone",
        "location",
        "district",
        "amount",
        "transaction_id",
        "account_number",
        "bank",
        "upi_id",
        "ifsc",
        "domain",
        "crypto_wallet",
        "url",
        "ip_address",
        "suspicious_domain",
        "suspicious_ip",
    ]

    # Deterministic fields ONLY from regex/gazetteer/validation.
    merged = dict(regex_data)
    # AI allowed ONLY for semantic fields.
    merged["crime_type"] = ai_data.get("crime_type", "Unknown")
    merged["severity"] = ai_data.get("severity", 4)
    merged["summary"] = ai_data.get("summary") or "No summary generated"

    present = sum(1 for k in key_fields if merged.get(k))
    merged["confidence_score"] = round(present / len(key_fields), 2) if key_fields else 0.0

    fingerprint_raw = "|".join(
        [
            str(merged.get("crime_type") or ""),
            str(merged.get("amount") or ""),
            str(merged.get("bank") or ""),
            str(merged.get("domain") or ""),
            str(merged.get("transaction_id") or ""),
            str(merged.get("account_number") or ""),
            str(merged.get("upi_id") or ""),
            str(merged.get("crypto_wallet") or ""),
            str(merged.get("ip_address") or ""),
            str(merged.get("district") or ""),
        ]
    )
    merged["fingerprint"] = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()

    return merged


@router.post("/extract", response_model=ExtractResponse)
def extract_complaint(payload: ExtractRequest, db: Session = Depends(get_db)):
    return _process_extraction(
        text=payload.text,
        db=db,
        officer_name=payload.officer_name,
        officer_id=payload.officer_id,
        station_name=payload.station_name,
    )


def _process_extraction(
    text: str,
    db: Session,
    officer_name: Optional[str] = None,
    officer_id: Optional[str] = None,
    station_name: Optional[str] = "Jaipur Cyber Cell",
    document_path: Optional[str] = None,
    ocr_confidence: Optional[float] = None,
    ocr_engine: Optional[str] = None,
):
    try:
        logger.info("Processing complaint extraction request")
        deterministic_data, field_confidence = deterministic_extract_with_confidence(text)
        contextual_entities, contextual_conf = parse_cyber_entities(text)
        taxonomy = classify_by_taxonomy(text)

        # Field-aware contextual overrides (deterministic -> contextual parser)
        for f in ["phone", "amount", "transaction_id", "account_number", "upi_id", "ifsc", "url", "date", "time"]:
            if contextual_entities.get(f):
                deterministic_data[f] = contextual_entities[f]
                field_confidence[f] = max(field_confidence.get(f, 0.0), contextual_conf.get(f, 0.0))

        # Multi-entity arrays for intelligence usage
        deterministic_data["phones"] = contextual_entities.get("phones", [])
        deterministic_data["accounts"] = contextual_entities.get("accounts", [])
        deterministic_data["transactions"] = contextual_entities.get("transactions", [])
        deterministic_data["urls"] = contextual_entities.get("urls", [])
        deterministic_data["upi_ids"] = contextual_entities.get("upi_ids", [])
        deterministic_data["ifsc_codes"] = contextual_entities.get("ifsc_codes", [])

        ner_data, ner_conf = ner_extract(text)

        # NER is allowed ONLY for soft fields (names/organizations), never for IDs/numbers/banks/districts.
        if not deterministic_data.get("victim_name") and ner_data.get("victim_name"):
            deterministic_data["victim_name"] = ner_data["victim_name"]
            field_confidence["victim_name"] = float(ner_conf.get("victim_name", 0.6))

        ai_data = enrich_with_ai(text, deterministic_data)
        merged_data = _merge_data(deterministic_data, ai_data)
        merged_data["crime_type_rule"] = taxonomy.get("crime_type_rule")
        merged_data["taxonomy_confidence"] = taxonomy.get("rule_confidence", 0.0)

        # Hybrid classification: taxonomy wins when confidence is high.
        if float(merged_data["taxonomy_confidence"]) >= 0.8:
            merged_data["crime_type"] = str(merged_data["crime_type_rule"])
            # Keep severity deterministic around rule estimate when rule is reliable.
            merged_data["severity"] = int(taxonomy.get("rule_severity", merged_data.get("severity", 4)))

        merged_data["field_confidence"] = field_confidence
        field_confidence_json = json.dumps(field_confidence, ensure_ascii=False)

        # Final schema validation gate before DB insertion.
        validated_response = ExtractResponse.model_validate(
            {
                **merged_data,
                "officer_name": officer_name,
                "officer_id": officer_id,
                "station_name": station_name,
                "status": "open",
                "document_path": document_path,
                "ocr_confidence": ocr_confidence,
                "ocr_engine": ocr_engine,
            }
        ).model_dump()

        complaint = Complaint(
            **{
                k: v
                for k, v in validated_response.items()
                if k
                not in {
                    "field_confidence",
                    "officer_name",
                    "officer_id",
                    "station_name",
                    "status",
                    "phones",
                    "accounts",
                    "transactions",
                    "urls",
                    "upi_ids",
                    "ifsc_codes",
                }
            },
            raw_text=text,
            officer_name=officer_name,
            officer_id=officer_id,
            station_name=station_name,
            field_confidence=field_confidence_json,
            phones=json.dumps(validated_response.get("phones", []), ensure_ascii=False),
            accounts=json.dumps(validated_response.get("accounts", []), ensure_ascii=False),
            transactions=json.dumps(validated_response.get("transactions", []), ensure_ascii=False),
            urls=json.dumps(validated_response.get("urls", []), ensure_ascii=False),
            upi_ids=json.dumps(validated_response.get("upi_ids", []), ensure_ascii=False),
            ifsc_codes=json.dumps(validated_response.get("ifsc_codes", []), ensure_ascii=False),
        )
        db.add(complaint)
        db.commit()
        logger.info("Complaint saved with id=%s", complaint.id)

        db.add(
            AuditLog(
                complaint_id=complaint.id,
                event_type="EXTRACTED",
                officer_name=officer_name,
                station_name=station_name,
                request_received_at=datetime.now(timezone.utc),
                details=f"confidence_score={complaint.confidence_score}, model_fields_filled={sum(1 for k,v in merged_data.items() if v)}",
            )
        )
        db.commit()

        return ExtractResponse.model_validate(
            {
                **validated_response,
                "status": complaint.status,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process complaint") from exc


@router.post("/extract/document", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile = File(...),
    officer_name: Optional[str] = Form(default=None),
    officer_id: Optional[str] = Form(default=None),
    station_name: Optional[str] = Form(default="Jaipur Cyber Cell"),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bmp", ".tif", ".tiff"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    docs_dir = get_documents_dir()
    safe_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:10]}{suffix}"
    saved_path = docs_dir / safe_name

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty uploaded file")
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        saved_path.write_bytes(data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to store uploaded document: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store uploaded document") from exc

    try:
        ocr_result = extract_text_from_document(saved_path)
    except OCRRuntimeError as exc:
        logger.error("OCR runtime failure: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    return _process_extraction(
        text=ocr_result["text"],
        db=db,
        officer_name=officer_name,
        officer_id=officer_id,
        station_name=station_name,
        document_path=str(saved_path),
        ocr_confidence=ocr_result.get("ocr_confidence"),
        ocr_engine=ocr_result.get("ocr_engine"),
    )


@router.get("/cases", response_model=list[CaseResponse])
def list_cases(db: Session = Depends(get_db)):
    try:
        cases = db.query(Complaint).order_by(Complaint.created_at.desc()).all()
        return cases
    except Exception as exc:
        logger.exception("Failed to fetch cases: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch cases") from exc
