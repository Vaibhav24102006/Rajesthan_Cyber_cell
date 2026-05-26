from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    complaint_id: Mapped[int] = mapped_column(Integer, ForeignKey("complaints.id"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)

    officer_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    station_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    request_received_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    details: Mapped[str] = mapped_column(Text, nullable=False, default="")

