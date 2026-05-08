from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
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
    crime_type: Mapped[str] = mapped_column(String(80), nullable=False, default="Unknown")
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="No summary generated")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Operational metadata
    officer_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    officer_id: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    station_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    # Intelligence fields
    district: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    upi_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    ifsc: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)
    domain: Mapped[Optional[str]] = mapped_column(String(220), nullable=True)
    crypto_wallet: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
    suspicious_domain: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    suspicious_ip: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Per-field confidence for deterministic extraction (stored as JSON string)
    field_confidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Multi-entity contextual intelligence (stored as JSON strings)
    phones: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    accounts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transactions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    urls: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    upi_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ifsc_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    crime_type_rule: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    taxonomy_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    document_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Raw complaint text")
    officer_name: Optional[str] = Field(default=None, description="Officer entering the complaint")
    officer_id: Optional[str] = Field(default=None, description="Badge / employee id")
    station_name: Optional[str] = Field(default="Jaipur Cyber Cell", description="Police station / unit name")


class ExtractResponse(BaseModel):
    victim_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    district: Optional[str] = None
    amount: Optional[str] = None
    transaction_id: Optional[str] = None
    account_number: Optional[str] = None
    bank: Optional[str] = None
    upi_id: Optional[str] = None
    ifsc: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    crypto_wallet: Optional[str] = None
    ip_address: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    crime_type: str
    severity: int
    summary: str
    confidence_score: float = 0.0
    fingerprint: Optional[str] = None
    officer_name: Optional[str] = None
    officer_id: Optional[str] = None
    station_name: Optional[str] = None
    status: str = "open"
    suspicious_domain: bool = False
    suspicious_ip: bool = False
    field_confidence: Dict[str, float] = Field(default_factory=dict)
    phones: list[dict] = Field(default_factory=list)
    accounts: list[dict] = Field(default_factory=list)
    transactions: list[dict] = Field(default_factory=list)
    urls: list[dict] = Field(default_factory=list)
    upi_ids: list[str] = Field(default_factory=list)
    ifsc_codes: list[str] = Field(default_factory=list)
    crime_type_rule: Optional[str] = None
    taxonomy_confidence: float = 0.0
    document_path: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_engine: Optional[str] = None

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        if v is None:
            return 4
        try:
            sv = int(float(str(v).strip()))
            return max(1, min(sv, 10))
        except (TypeError, ValueError):
            return 4

    @field_validator("crime_type", mode="before")
    @classmethod
    def _coerce_crime_type(cls, v):
        if v is None:
            return "Unknown"
        s = str(v).strip()
        return s if s else "Unknown"

    @field_validator("summary", mode="before")
    @classmethod
    def _coerce_summary(cls, v):
        if v is None:
            return "No summary generated"
        s = str(v).strip()
        return s if s else "No summary generated"

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        if v is None:
            return "open"
        s = str(v).strip()
        return s if s else "open"

    @field_validator("suspicious_domain", "suspicious_ip", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "y"}

    @field_validator("phones", "accounts", "transactions", "urls", mode="before")
    @classmethod
    def _parse_obj_array(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        try:
            import json

            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @field_validator("upi_ids", "ifsc_codes", mode="before")
    @classmethod
    def _parse_str_array(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        try:
            import json

            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return []
        except Exception:
            return []

    @field_validator("taxonomy_confidence", mode="before")
    @classmethod
    def _coerce_tax_conf(cls, v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    @field_validator("ocr_confidence", mode="before")
    @classmethod
    def _coerce_ocr_conf(cls, v):
        if v is None:
            return None
        try:
            fv = float(v)
            return max(0.0, min(fv, 1.0))
        except (TypeError, ValueError):
            return None


class CaseResponse(ExtractResponse):
    id: int
    raw_text: str
    created_at: datetime

    class Config:
        from_attributes = True


    @field_validator("field_confidence", mode="before")
    @classmethod
    def _parse_field_confidence(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        try:
            import json

            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
