from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.complaint import CaseResponse, Complaint


router = APIRouter()


def _match_score(a: Any, b: Any) -> int:
    if a is None or b is None:
        return 0
    return 2 if str(a).strip().lower() == str(b).strip().lower() else 0


def _amount_similarity(a: Optional[str], b: Optional[str]) -> int:
    if not a or not b:
        return 0
    try:
        fa = float(a)
        fb = float(b)
    except ValueError:
        return 0
    if fa == 0:
        return 0
    rel = abs(fa - fb) / abs(fa)
    return 1 if rel <= 0.1 else 0


def _compute_similarity(base: Complaint, other: Complaint) -> int:
    score = 0
    score += 2 if base.crime_type == other.crime_type else 0
    score += _match_score(base.bank, other.bank)
    score += _match_score(base.domain, other.domain)

    # Strong identifiers
    if base.transaction_id and other.transaction_id and str(base.transaction_id).strip() == str(other.transaction_id).strip():
        score += 3
    if base.upi_id and other.upi_id and str(base.upi_id).strip().lower() == str(other.upi_id).strip().lower():
        score += 3
    score += _amount_similarity(base.amount, other.amount)
    score += _match_score(base.crypto_wallet, other.crypto_wallet)
    score += _match_score(base.ip_address, other.ip_address)
    score += 1 if base.district and other.district and base.district == other.district else 0
    return score


@router.get("/cases/{case_id}/similar")
def similar_cases(case_id: int, limit: int = Query(default=5, ge=1, le=20), db: Session = Depends(get_db)) -> Dict[str, Any]:
    base = db.query(Complaint).filter(Complaint.id == case_id).first()
    if not base:
        raise HTTPException(status_code=404, detail="Case not found")

    all_cases = db.query(Complaint).filter(Complaint.id != case_id).all()
    scored: List[Tuple[int, Complaint]] = [(_compute_similarity(base, c), c) for c in all_cases]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]

    results = []
    for score, c in top:
        results.append(
            {
                "case_id": c.id,
                "similarity_score": int(score),
                "case": CaseResponse.model_validate(c).model_dump(),
            }
        )
    return {"base_case_id": case_id, "results": results}

