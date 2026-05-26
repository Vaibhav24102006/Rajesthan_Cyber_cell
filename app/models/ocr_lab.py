from typing import Dict, Any, Optional
from pydantic import BaseModel

class OCRLabRequest(BaseModel):
    # Depending on how frontend sends it, we might just use Form data for file upload,
    # but we can keep a schema if needed for text-only lab testing.
    text: str

class OCRLabResponse(BaseModel):
    raw_text: str
    corrected_text: str
    reconstructed_text: str = ""
    final_readable_text: str = ""
    reconstruction_confidence_score: float = 0.0
    reconstruction_source: str = "unknown"
    detected_domain: Optional[str] = "unknown"
    reconstruction_mode: Optional[str] = "balanced_reconstruction"
    
    # Multilingual & Dialect Redesign fields
    detected_language: Optional[str] = "unknown"
    detected_dialect: Optional[str] = "unknown"
    dialect_detection: Dict[str, Any] = {}
    routing_decision: Optional[str] = "unknown"
    normalized_dialect_text: Optional[str] = ""
    normalized_regional_text: Optional[str] = ""
    translated_text: Optional[str] = ""
    translation_output: Optional[str] = ""
    structured_english_complaint: Optional[str] = ""
    entity_preservation_audit: Dict[str, Any] = {}
    language_route_confidence: float = 0.0
    language_warnings: list = []
    stage_confidence: Dict[str, float] = {}
    benchmarks: Dict[str, Any] = {}
    
    raw_confidence: Optional[float] = None
    corrected_confidence: float = 0.0
    
    # Deterministic entities from the final (accepted) text
    extracted_entities: Dict[str, Any]
    
    # Audit trail details
    was_rejected: bool = False
    rejection_reason: Optional[str] = None
    
    # Advanced OCR metadata
    uncertain_regions: list = []
    officer_review_regions: list = []
    line_results: list = []
    preprocessing_metadata: dict = {}
    processing_time: float = 0.0
    file_id: str = ""
    correction_status: str = "not_started"
    timeout_detected: bool = False
    fallback_used: bool = False
    correction_model: str = "phi3"
    active_ocr_engine: str = "unknown"
    guardrail_status: str = "not_checked"
    lightweight_mode: bool = False
    lightweight_reason: str = ""
    performance_metrics: Dict[str, Any] = {}
    stage_timings: Dict[str, Any] = {}
