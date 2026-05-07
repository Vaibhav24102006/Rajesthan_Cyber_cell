from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class Complaint(Base):
    __tablename__ = "complaints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    victim_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(180), nullable=True)
    amount: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    transaction_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    account_number: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    bank: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    crime_type: Mapped[str] = mapped_column(String(80), nullable=False, default="Other")
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Raw complaint text")


class ExtractResponse(BaseModel):
    victim_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    amount: Optional[str] = None
    transaction_id: Optional[str] = None
    account_number: Optional[str] = None
    bank: Optional[str] = None
    crime_type: str
    severity: int
    summary: str


class CaseResponse(ExtractResponse):
    id: int
    raw_text: str
    created_at: datetime

    class Config:
        from_attributes = True
