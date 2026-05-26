import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy import Float, case, cast, func, desc
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.complaint import Complaint


router = APIRouter(prefix="/analytics")


def _parse_amount_sum(db: Session) -> float:
    # amount is stored as numeric string; cast in SQLite.
    total = db.query(func.sum(cast(Complaint.amount, Float))).scalar()
    return float(total or 0.0)


@router.get("/summary")
def analytics_summary(db: Session = Depends(get_db)) -> Dict:
    total_cases = db.query(func.count(Complaint.id)).scalar() or 0
    open_cases = db.query(func.count(Complaint.id)).filter(Complaint.status == "open").scalar() or 0
    resolved_cases = db.query(func.count(Complaint.id)).filter(Complaint.status == "resolved").scalar() or 0

    avg_sev = db.query(func.avg(cast(Complaint.severity, Float))).scalar()
    avg_sev_val = float(avg_sev) if avg_sev is not None else 0.0

    total_amount = _parse_amount_sum(db)

    high_sev_count = (
        db.query(func.count(Complaint.id)).filter(Complaint.severity >= 8).scalar() or 0
    )

    # Top crimes
    top_crimes_rows = (
        db.query(Complaint.crime_type, func.count(Complaint.id).label("n"))
        .group_by(Complaint.crime_type)
        .order_by(desc("n"))
        .limit(6)
        .all()
    )
    top_crimes = [{"crime_type": r[0], "count": int(r[1])} for r in top_crimes_rows]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": int(total_cases),
        "open_cases": int(open_cases),
        "resolved_cases": int(resolved_cases),
        "avg_severity": round(avg_sev_val, 2),
        "total_amount_at_risk": round(total_amount, 2),
        "high_severity_count": int(high_sev_count),
        "top_crimes": top_crimes,
    }


@router.get("/banks")
def analytics_banks(db: Session = Depends(get_db)) -> List[Dict]:
    rows = (
        db.query(Complaint.bank, func.count(Complaint.id).label("n"))
        .filter(Complaint.bank.isnot(None))
        .group_by(Complaint.bank)
        .order_by(desc("n"))
        .limit(10)
        .all()
    )
    return [{"bank": r[0], "count": int(r[1])} for r in rows if r[0]]


@router.get("/districts")
def analytics_districts(db: Session = Depends(get_db)) -> List[Dict]:
    rows = (
        db.query(Complaint.district, func.count(Complaint.id).label("n"))
        .filter(Complaint.district.isnot(None))
        .group_by(Complaint.district)
        .order_by(desc("n"))
        .limit(10)
        .all()
    )
    return [{"district": r[0], "count": int(r[1])} for r in rows if r[0]]


@router.get("/trends")
def analytics_trends(days: int = 14, db: Session = Depends(get_db)) -> Dict:
    days = max(1, min(int(days), 60))
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days - 1)

    # Total counts per day
    rows = (
        db.query(func.date(Complaint.created_at).label("d"), func.count(Complaint.id).label("n"))
        .filter(Complaint.created_at >= start)
        .group_by(func.date(Complaint.created_at))
        .order_by(func.date(Complaint.created_at))
        .all()
    )
    counts_by_day = {str(r[0]): int(r[1]) for r in rows if r[0] is not None}

    points = []
    for i in range(days):
        day = start + timedelta(days=i)
        day_s = day.isoformat()
        points.append({"date": day_s, "total": counts_by_day.get(day_s, 0)})

    # Also provide top crime type for the period (useful for legend)
    top_crime_rows = (
        db.query(Complaint.crime_type, func.count(Complaint.id).label("n"))
        .filter(Complaint.created_at >= start)
        .group_by(Complaint.crime_type)
        .order_by(desc("n"))
        .limit(5)
        .all()
    )
    top_crimes = [{"crime_type": r[0], "count": int(r[1])} for r in top_crime_rows if r[0]]

    return {
        "days": days,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "points": points,
        "top_crimes": top_crimes,
    }

