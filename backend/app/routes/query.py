import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Float, cast, desc, func, or_
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.complaint import CaseResponse, Complaint


router = APIRouter()


BANK_KEYWORDS = {
    "canara": "Canara Bank",
    "sbi": "SBI",
    "state bank": "SBI",
    "hdfc": "HDFC",
    "axis": "Axis Bank",
    "icici": "ICICI",
    "kotak": "Kotak",
    "bo b": "Bank of Baroda",
    "bank of baroda": "Bank of Baroda",
    "union": "Union Bank",
}

DISTRICTS_RAJASTHAN = [
    "Jaipur",
    "Jodhpur",
    "Ajmer",
    "Udaipur",
    "Kota",
    "Bikaner",
    "Alwar",
    "Sikar",
    "Nagaur",
    "Jhunjhunu",
    "Churu",
    "Dausa",
    "Dholpur",
    "Bharatpur",
    "Sawai Madhopur",
    "Karauli",
    "Baran",
    "Jhalawar",
    "Chittorgarh",
    "Bhilwara",
    "Pratapgarh",
    "Rajsamand",
    "Sirohi",
    "Tonk",
    "Jaisalmer",
]

CRIME_KEYWORDS = {
    "phishing": "Phishing",
    "fake link": "Phishing",
    "fraud": "Financial Fraud",
    "financial fraud": "Financial Fraud",
    "ransomware": "Ransomware",
    "hacking": "Hacking",
    "identity theft": "Identity Theft",
    "blackmail": "Harassment",
    "harassment": "Harassment",
    "stalking": "Cyber Stalking",
}


def _parse_amount(query: str) -> Optional[float]:
    # Example patterns: "above ₹50,000" / "over 50000 rupees"
    m = re.search(r"(?:above|over|more than|greater than)\s*(?:₹|rs\.?|rupees?)?\s*([0-9][0-9,]*)", query, flags=re.IGNORECASE)
    if not m:
        # If no explicit threshold keyword, allow "₹50000" extraction but do not force it.
        m = re.search(r"(?:₹|rs\.?|rupees?)\s*([0-9][0-9,]*)", query, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_last_days(query: str) -> Optional[int]:
    m = re.search(r"last\s+(\d+)\s+day", query, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"past\s+(\d+)\s+day", query, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_crime_type(query: str) -> Optional[str]:
    q = query.lower()
    for k, v in CRIME_KEYWORDS.items():
        if k in q:
            return v
    return None


def _extract_district(query: str) -> Optional[str]:
    q = query.lower()
    for d in DISTRICTS_RAJASTHAN:
        if d.lower() in q:
            return d
    return None


def _extract_bank(query: str) -> Optional[str]:
    q = query.lower()
    for k, v in BANK_KEYWORDS.items():
        if k in q:
            return v
    return None


def _safe_build_sql_description(filters: Dict[str, Any]) -> str:
    # Human-readable only. We never execute raw SQL.
    where = []
    if "crime_type" in filters:
        where.append(f"crime_type = '{filters['crime_type']}'")
    if "bank" in filters:
        where.append(f"bank ILIKE '%{filters['bank']}%'")
    if "district" in filters:
        where.append(f"district = '{filters['district']}'")
    if "min_amount" in filters:
        where.append(f"CAST(amount AS FLOAT) >= {filters['min_amount']}")
    if "start_date" in filters:
        where.append(f"created_at >= '{filters['start_date']}'")
    return "SELECT * FROM complaints" + ((" WHERE " + " AND ".join(where)) if where else "")


@router.post("/query")
def natural_language_query(payload: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    limit = int(payload.get("limit", 50) or 50)
    limit = max(1, min(limit, 200))

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    ql = query.lower()
    district_wise = "district-wise" in ql or "district wise" in ql

    crime_type = _extract_crime_type(query)
    bank = _extract_bank(query)
    district = _extract_district(query)
    days = _extract_last_days(query)
    min_amount = _parse_amount(query)

    start_date = None
    if days is not None:
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=timezone.utc)

    # Build filters using SQLAlchemy (safe, parameterized)
    filters: Dict[str, Any] = {
        "crime_type": crime_type,
        "bank": bank,
        "district": district,
        "min_amount": min_amount,
        "start_date": start_date.isoformat() if start_date else None,
    }
    sql_desc = _safe_build_sql_description({k: v for k, v in filters.items() if v is not None})

    base_q = db.query(Complaint)

    if crime_type:
        base_q = base_q.filter(Complaint.crime_type == crime_type)
    if bank:
        base_q = base_q.filter(Complaint.bank.isnot(None)).filter(Complaint.bank.ilike(f"%{bank}%"))
    if district and not district_wise:
        base_q = base_q.filter(Complaint.district == district)
    if min_amount is not None and "fraud" in ql:
        # If intent includes "fraud" threshold, apply amount filter.
        base_q = base_q.filter(cast(Complaint.amount, Float) >= float(min_amount))
    if start_date:
        base_q = base_q.filter(Complaint.created_at >= start_date)

    if district_wise:
        if not crime_type:
            # For district-wise queries without explicit crime type, return district counts for all cases.
            rows = base_q.with_entities(Complaint.district, func.count(Complaint.id).label("n")).filter(
                Complaint.district.isnot(None)
            ).group_by(Complaint.district).order_by(desc("n")).limit(20).all()
        else:
            rows = base_q.with_entities(Complaint.district, func.count(Complaint.id).label("n")).filter(
                Complaint.district.isnot(None)
            ).group_by(Complaint.district).order_by(desc("n")).limit(20).all()

        return {
            "type": "district_counts",
            "sql": sql_desc,
            "results": [{"district": r[0], "count": int(r[1])} for r in rows if r[0]],
        }

    cases = base_q.order_by(Complaint.severity.desc()).limit(limit).all()
    return {
        "type": "cases",
        "sql": sql_desc,
        "count": len(cases),
        "results": [CaseResponse.model_validate(c).model_dump() for c in cases],
    }

