from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.models.audit_log import AuditLog


router = APIRouter()


@router.get("/cases/{case_id}/audit")
def list_case_audit(case_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.complaint_id == case_id)
        .order_by(AuditLog.request_received_at.desc())
        .limit(50)
        .all()
    )
    return {
        "case_id": case_id,
        "logs": [
            {
                "event_type": l.event_type,
                "officer_name": l.officer_name,
                "station_name": l.station_name,
                "request_received_at": l.request_received_at.isoformat() if l.request_received_at else None,
                "details": l.details,
            }
            for l in logs
        ],
    }

