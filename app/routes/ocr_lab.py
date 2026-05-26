import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.database.db import get_db
from app.models.audit_log import OCRAuditLog
from app.models.ocr_lab import OCRLabResponse
from app.services.extractor import deterministic_extract_with_confidence
from app.services.ocr.dataset_manager import get_dataset_manager
from app.services.ocr.paddle_engine import paddle_engine
from app.services.ocr_correction import (
    correct_ocr_text,
    get_active_correction_model,
    is_ollama_reachable,
)
from app.services.indus_reconstruction_service import (
    reconstruct_ocr_text as reconstruct_ocr_text_with_indus,
    is_indus_reachable,
)
from app.services.ocr_service import (
    clean_ocr_text,
    extract_text_from_document,
    get_documents_dir,
    get_tesseract_runtime_diagnostics,
    is_paddle_available,
    is_tesseract_available,
    OCRRuntimeError,
    _resolve_tesseract_cmd,
)

logger = logging.getLogger(__name__)
router = APIRouter()
OCR_PIPELINE_MODE = "low_memory_cpu"
LIGHTWEIGHT_MODE_POLICY = os.getenv("OCR_LIGHTWEIGHT_MODE", "auto").strip().lower()
if LIGHTWEIGHT_MODE_POLICY not in {"auto", "on", "off"}:
    LIGHTWEIGHT_MODE_POLICY = "auto"
try:
    LIGHTWEIGHT_FILE_SIZE_MB_THRESHOLD = float(os.getenv("OCR_LIGHTWEIGHT_FILE_SIZE_MB_THRESHOLD", "3.0"))
except Exception:
    LIGHTWEIGHT_FILE_SIZE_MB_THRESHOLD = 3.0
try:
    LIGHTWEIGHT_OCR_RUNTIME_MS_THRESHOLD = float(os.getenv("OCR_LIGHTWEIGHT_OCR_RUNTIME_MS_THRESHOLD", "15000"))
except Exception:
    LIGHTWEIGHT_OCR_RUNTIME_MS_THRESHOLD = 15000.0
try:
    LIGHTWEIGHT_MEMORY_PERCENT_THRESHOLD = float(os.getenv("OCR_LIGHTWEIGHT_MEMORY_PERCENT_THRESHOLD", "85"))
except Exception:
    LIGHTWEIGHT_MEMORY_PERCENT_THRESHOLD = 85.0
try:
    LIGHTWEIGHT_MIN_AVAILABLE_GB = float(os.getenv("OCR_LIGHTWEIGHT_MIN_AVAILABLE_GB", "1.2"))
except Exception:
    LIGHTWEIGHT_MIN_AVAILABLE_GB = 1.2

_runtime_metrics_lock = threading.Lock()
_runtime_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "lightweight_runs": 0,
    "timeout_runs": 0,
    "fallback_runs": 0,
    "rejected_runs": 0,
    "totals_ms": {
        "ocr_runtime_ms": 0.0,
        "correction_runtime_ms": 0.0,
        "extraction_runtime_ms": 0.0,
        "total_pipeline_runtime_ms": 0.0,
    },
    "last_request": {},
}


class VerifyOCRRequest(BaseModel):
    file_id: str
    verified_text: str
    line_results: list


def _pick_active_engine(engine_usage: Dict[str, Any]) -> str:
    if not isinstance(engine_usage, dict) or not engine_usage:
        return "unknown"
    normalized = {str(k): int(v) for k, v in engine_usage.items() if isinstance(v, (int, float))}
    return max(normalized, key=normalized.get) if normalized else "unknown"


def _should_use_lightweight_mode(file_path: Path, ocr_runtime_ms: float) -> tuple[bool, str]:
    if LIGHTWEIGHT_MODE_POLICY == "on":
        return True, "forced_policy=on"
    if LIGHTWEIGHT_MODE_POLICY == "off":
        return False, "forced_policy=off"
    reasons = []
    file_size_mb = (file_path.stat().st_size / (1024 * 1024)) if file_path.exists() else 0.0
    if file_size_mb >= LIGHTWEIGHT_FILE_SIZE_MB_THRESHOLD:
        reasons.append(f"large_file={file_size_mb:.2f}MB")
    if ocr_runtime_ms >= LIGHTWEIGHT_OCR_RUNTIME_MS_THRESHOLD:
        reasons.append(f"slow_ocr={ocr_runtime_ms:.0f}ms")
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        if mem.percent >= LIGHTWEIGHT_MEMORY_PERCENT_THRESHOLD or mem.available < (LIGHTWEIGHT_MIN_AVAILABLE_GB * 1024 * 1024 * 1024):
            reasons.append(f"high_memory_pressure={mem.percent:.1f}%")
    except Exception:
        pass
    return (True, "; ".join(reasons)) if reasons else (False, "")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


class StageProfiler:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.timings: Dict[str, Any] = {}

    def start(self, stage: str) -> float:
        logger.info("[TRACE %s] [START] %s", self.trace_id, stage)
        return perf_counter()

    def end(self, stage: str, started_at: float, **metadata: Any) -> None:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        row = {"duration_ms": elapsed_ms}
        row.update(metadata)
        self.timings[stage] = row
        logger.info("[TRACE %s] [END] %s duration_ms=%.2f metadata=%s", self.trace_id, stage, elapsed_ms, metadata)


def _record_runtime_metrics(
    *,
    performance_metrics: Dict[str, Any],
    correction_meta: Dict[str, Any],
    active_engine: str,
    lightweight_mode: bool,
    lightweight_reason: str,
) -> None:
    with _runtime_metrics_lock:
        _runtime_metrics["total_requests"] += 1
        if lightweight_mode:
            _runtime_metrics["lightweight_runs"] += 1
        if bool(correction_meta.get("timeout_detected", False)):
            _runtime_metrics["timeout_runs"] += 1
        if bool(correction_meta.get("fallback_used", False)):
            _runtime_metrics["fallback_runs"] += 1
        if str(correction_meta.get("correction_status", "")).startswith("rejected"):
            _runtime_metrics["rejected_runs"] += 1

        totals = _runtime_metrics["totals_ms"]
        totals["ocr_runtime_ms"] += _safe_float(performance_metrics.get("ocr_runtime_ms", 0.0))
        totals["correction_runtime_ms"] += _safe_float(performance_metrics.get("correction_runtime_ms", 0.0))
        totals["extraction_runtime_ms"] += _safe_float(performance_metrics.get("extraction_runtime_ms", 0.0))
        totals["total_pipeline_runtime_ms"] += _safe_float(performance_metrics.get("total_pipeline_runtime_ms", 0.0))

        _runtime_metrics["last_request"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "active_ocr_engine": active_engine,
            "correction_model": correction_meta.get("correction_model", get_active_correction_model()),
            "correction_status": correction_meta.get("correction_status", "unknown"),
            "guardrail_status": correction_meta.get("guardrail_status", "not_checked"),
            "timeout_detected": bool(correction_meta.get("timeout_detected", False)),
            "fallback_used": bool(correction_meta.get("fallback_used", False)),
            "lightweight_mode": lightweight_mode,
            "lightweight_reason": lightweight_reason,
            "performance_metrics": {
                "ocr_runtime_ms": round(_safe_float(performance_metrics.get("ocr_runtime_ms", 0.0)), 2),
                "correction_runtime_ms": round(_safe_float(performance_metrics.get("correction_runtime_ms", 0.0)), 2),
                "extraction_runtime_ms": round(_safe_float(performance_metrics.get("extraction_runtime_ms", 0.0)), 2),
                "total_pipeline_runtime_ms": round(_safe_float(performance_metrics.get("total_pipeline_runtime_ms", 0.0)), 2),
            },
        }


def _get_runtime_metrics_snapshot() -> Dict[str, Any]:
    with _runtime_metrics_lock:
        total_requests = int(_runtime_metrics.get("total_requests", 0))
        totals_ms = dict(_runtime_metrics.get("totals_ms", {}))
        avg_metrics = {
            "ocr_runtime_ms": round((totals_ms.get("ocr_runtime_ms", 0.0) / total_requests), 2) if total_requests else 0.0,
            "correction_runtime_ms": round((totals_ms.get("correction_runtime_ms", 0.0) / total_requests), 2) if total_requests else 0.0,
            "extraction_runtime_ms": round((totals_ms.get("extraction_runtime_ms", 0.0) / total_requests), 2) if total_requests else 0.0,
            "total_pipeline_runtime_ms": round((totals_ms.get("total_pipeline_runtime_ms", 0.0) / total_requests), 2) if total_requests else 0.0,
        }
        return {
            "total_requests": total_requests,
            "lightweight_runs": int(_runtime_metrics.get("lightweight_runs", 0)),
            "timeout_runs": int(_runtime_metrics.get("timeout_runs", 0)),
            "fallback_runs": int(_runtime_metrics.get("fallback_runs", 0)),
            "rejected_runs": int(_runtime_metrics.get("rejected_runs", 0)),
            "average_runtime_ms": avg_metrics,
            "last_request": dict(_runtime_metrics.get("last_request", {})),
        }


@router.get("/health")
def ocr_lab_health() -> Dict[str, Any]:
    paddle_available = is_paddle_available()
    tesseract_available = is_tesseract_available()
    ollama_available = is_ollama_reachable()
    indus_available = is_indus_reachable()
    correction_model = get_active_correction_model()
    pipeline_ready = paddle_available or tesseract_available
    resolved_tesseract_cmd = _resolve_tesseract_cmd()
    metrics_snapshot = _get_runtime_metrics_snapshot()
    return {
        "paddleocr_available": paddle_available,
        "tesseract_available": tesseract_available,
        "tesseract_cmd": resolved_tesseract_cmd,
        "tesseract_cmd_env": os.getenv("TESSERACT_CMD"),
        "ollama_reachable": ollama_available,
        "indus_reachable": indus_available,
        "indus_api_url_configured": bool(os.getenv("INDUS_API_URL")),
        "indus_health_url": os.getenv("INDUS_HEALTH_URL"),
        "indus_model": os.getenv("INDUS_MODEL", "handwriting-normalization"),
        "correction_model_active": correction_model,
        "ocr_pipeline_readiness": pipeline_ready,
        "tesseract_runtime": get_tesseract_runtime_diagnostics(),
        "ocr_pipeline_mode": OCR_PIPELINE_MODE,
        "lightweight_mode_policy": LIGHTWEIGHT_MODE_POLICY,
        "lightweight_thresholds": {
            "file_size_mb": LIGHTWEIGHT_FILE_SIZE_MB_THRESHOLD,
            "ocr_runtime_ms": LIGHTWEIGHT_OCR_RUNTIME_MS_THRESHOLD,
            "memory_percent": LIGHTWEIGHT_MEMORY_PERCENT_THRESHOLD,
            "min_available_gb": LIGHTWEIGHT_MIN_AVAILABLE_GB,
        },
        "runtime_metrics": metrics_snapshot,
    }


@router.get("/metrics")
def ocr_lab_metrics() -> Dict[str, Any]:
    return {
        "ocr_pipeline_mode": OCR_PIPELINE_MODE,
        "lightweight_mode_policy": LIGHTWEIGHT_MODE_POLICY,
        "runtime_metrics": _get_runtime_metrics_snapshot(),
    }

@router.post("/free-memory")
async def free_memory_endpoint():
    try:
        paddle_engine.free_memory()
        return {"status": "success", "message": "OCR model memory released"}
    except Exception as e:
        logger.error("Failed to free memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify")
async def verify_ocr_endpoint(req: VerifyOCRRequest):
    try:
        dataset_mgr = get_dataset_manager()
        storage_dir = Path(__file__).resolve().parents[3] / "storage" / "documents"
        img_path = storage_dir / req.file_id
        dataset_mgr.save_sample(
            raw_img_path=img_path,
            ocr_results={"raw_text": req.verified_text, "line_results": req.line_results},
            human_verified=True,
            verified_text=req.verified_text,
        )

        corrected_text, _, _, _ = correct_ocr_text(req.verified_text)
        entities = deterministic_extract_with_confidence(corrected_text)
        return {
            "status": "success",
            "human_audited": True,
            "corrected_text": corrected_text,
            "extracted_entities": entities,
        }
    except Exception as e:
        logger.error("Failed verification: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/report")
def ocr_evaluation_report(db: Session = Depends(get_db)) -> Dict[str, Any]:
    required_entity_keys = ["phone", "account_number", "bank", "amount", "transaction_id", "upi_id", "ifsc"]
    logs = db.query(OCRAuditLog).order_by(OCRAuditLog.created_at.desc()).all()
    total_logs = len(logs)
    accepted = [log for log in logs if not log.was_rejected]
    rejected = [log for log in logs if log.was_rejected]

    per_sample_coverage = []
    missing_entity_counts: Dict[str, int] = {k: 0 for k in required_entity_keys}
    report_rows = []
    for log in logs[:100]:
        try:
            corr_entities = json.loads(log.corrected_entities_json or "{}")
        except Exception:
            corr_entities = {}
        present = sum(1 for k in required_entity_keys if corr_entities.get(k))
        coverage = (present / len(required_entity_keys)) if required_entity_keys else 0.0
        per_sample_coverage.append(coverage)
        for key in required_entity_keys:
            if not corr_entities.get(key):
                missing_entity_counts[key] += 1
        report_rows.append(
            {
                "created_at": str(log.created_at),
                "correction_accepted": not bool(log.was_rejected),
                "rejection_reason": log.rejection_reason,
                "entity_coverage": round(coverage, 4),
            }
        )

    extraction_accuracy = sum(per_sample_coverage) / len(per_sample_coverage) if per_sample_coverage else 0.0
    correction_acceptance_rate = (len(accepted) / total_logs) if total_logs else 0.0

    dataset_path = Path(__file__).resolve().parents[3] / "storage" / "datasets" / "ocr_validation" / "dataset.jsonl"
    confidence_values = []
    if dataset_path.exists():
        try:
            for line in dataset_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                confidence_values.append(float(row.get("confidence", 0.0)))
        except Exception:
            pass
    avg_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

    return {
        "summary": {
            "total_samples": total_logs,
            "extraction_accuracy": round(extraction_accuracy, 4),
            "confidence": round(avg_conf, 4),
            "correction_acceptance_rate": round(correction_acceptance_rate, 4),
            "rejected_corrections": len(rejected),
        },
        "missing_entities": missing_entity_counts,
        "report_rows": report_rows,
    }


@router.post("/extract/document", response_model=OCRLabResponse)
async def ocr_lab_extract_document(
    file: UploadFile = File(...),
    reconstruction_mode: str = "structured_english_output",
    document_domain: str = "auto",
    db: Session = Depends(get_db),
):
    pipeline_start = perf_counter()
    trace_id = uuid4().hex[:10]
    profiler = StageProfiler(trace_id)
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bmp", ".tif", ".tiff"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    docs_dir = get_documents_dir()
    safe_name = f"lab_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:10]}{suffix}"
    saved_path = docs_dir / safe_name

    try:
        stage_start = profiler.start("upload_read")
        data = await file.read()
        profiler.end("upload_read", stage_start, size_bytes=len(data) if data else 0)
        if not data:
            raise HTTPException(status_code=400, detail="Empty uploaded file")
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        stage_start = profiler.start("upload_save")
        saved_path.write_bytes(data)
        profiler.end("upload_save", stage_start, path=saved_path.name)
    except Exception as exc:
        logger.exception("Failed to store uploaded document: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store uploaded document") from exc

    try:
        stage_start = profiler.start("ocr_extraction")
        ocr_result = await run_in_threadpool(extract_text_from_document, saved_path)
        profiler.end("ocr_extraction", stage_start)
        ocr_runtime_ms = profiler.timings["ocr_extraction"]["duration_ms"]
        raw_text = clean_ocr_text(ocr_result["text"])
        if not raw_text.strip():
            logger.error("[TRACE %s] OCR extraction returned empty text. Halting pipeline.", trace_id)
            raise HTTPException(
                status_code=422,
                detail="OCR extraction returned empty text. Multilingual complaint ingestion pipeline halted."
            )
        raw_confidence = ocr_result.get("ocr_confidence", 0.0)
    except OCRRuntimeError as exc:
        logger.error("OCR runtime failure: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    extraction_start = perf_counter()
    stage_start = profiler.start("raw_entity_extraction")
    raw_entities, _ = await run_in_threadpool(deterministic_extract_with_confidence, raw_text)
    profiler.end("raw_entity_extraction", stage_start)

    lightweight_mode, lightweight_reason = _should_use_lightweight_mode(saved_path, ocr_runtime_ms)
    reconstructed_text = raw_text
    was_rejected = False
    rejection_reason = None
    reconstruction_meta: Dict[str, Any] = {
        "reconstruction_model": get_active_correction_model(),
        "reconstruction_status": "skipped_lightweight_mode" if lightweight_mode else "not_started",
        "reconstruction_source": "lightweight_skipped" if lightweight_mode else "not_started",
        "reconstruction_confidence_score": 0.0,
        "reconstruction_runtime_ms": 0.0,
        "timeout_detected": False,
        "fallback_used": bool(lightweight_mode),
        "guardrail_status": "skipped" if lightweight_mode else "not_checked",
        "detected_domain": "unknown",
        "detected_language": "unknown",
        "detected_dialect": "unknown",
        "dialect_detection": {},
        "routing_decision": "unknown",
        "normalized_dialect_text": "",
        "normalized_regional_text": "",
        "translated_text": "",
        "translation_output": "",
        "structured_english_complaint": "",
        "entity_preservation_audit": {},
        "language_route_confidence": 0.0,
        "language_warnings": [],
        "reconstruction_mode": reconstruction_mode,
    }
    if lightweight_mode:
        logger.warning(
            "[TRACE %s] lightweight policy active (%s); language router will still run with deterministic fallbacks.",
            trace_id,
            lightweight_reason,
        )
    if raw_text.strip():
        stage_start = profiler.start("reconstruction")
        reconstructed_text, was_rejected, rejection_reason, reconstruction_meta = await run_in_threadpool(
            reconstruct_ocr_text_with_indus,
            raw_text,
            ocr_confidence=raw_confidence,
            reconstruction_mode=reconstruction_mode,
            document_domain=document_domain,
        )
        profiler.end(
            "reconstruction",
            stage_start,
            status=reconstruction_meta.get("reconstruction_status", "unknown"),
            source=reconstruction_meta.get("reconstruction_source", "unknown"),
        )
    else:
        reconstruction_meta["reconstruction_status"] = "skipped_empty_ocr"
        reconstruction_meta["reconstruction_source"] = "empty_ocr"
        reconstruction_meta["fallback_used"] = True
        reconstruction_meta["detected_domain"] = "unknown"
        reconstruction_meta["reconstruction_mode"] = reconstruction_mode

    stage_start = profiler.start("response_serialization")
    final_text = reconstructed_text
    corrected_entities, corrected_conf_dict = await run_in_threadpool(deterministic_extract_with_confidence, final_text)
    extraction_runtime_ms = round((perf_counter() - extraction_start) * 1000, 2)
    valid_confs = [v for k, v in corrected_conf_dict.items() if corrected_entities.get(k)]
    corrected_confidence = sum(valid_confs) / len(valid_confs) if valid_confs else raw_confidence

    pre_meta = dict(ocr_result.get("preprocessing_metadata", {}))
    active_engine = _pick_active_engine(pre_meta.get("engine_usage", {}))
    pre_meta["active_ocr_engine"] = active_engine
    logger.info("active OCR engine used per request: %s", active_engine)
    profiler.end("response_serialization", stage_start, active_ocr_engine=active_engine)

    total_pipeline_runtime_ms = round((perf_counter() - pipeline_start) * 1000, 2)
    performance_metrics = {
        "ocr_runtime_ms": ocr_runtime_ms,
        "correction_runtime_ms": reconstruction_meta.get("reconstruction_runtime_ms", 0.0),
        "extraction_runtime_ms": extraction_runtime_ms,
        "total_pipeline_runtime_ms": total_pipeline_runtime_ms,
        "active_ocr_engine": active_engine,
    }
    _record_runtime_metrics(
        performance_metrics=performance_metrics,
        correction_meta=reconstruction_meta,
        active_engine=active_engine,
        lightweight_mode=lightweight_mode,
        lightweight_reason=lightweight_reason,
    )

    try:
        audit_log = OCRAuditLog(
            raw_text=raw_text,
            corrected_text=final_text,
            raw_entities_json=json.dumps(raw_entities),
            corrected_entities_json=json.dumps(corrected_entities),
            was_rejected=was_rejected,
            rejection_reason=rejection_reason,
        )
        db.add(audit_log)
        db.commit()
    except Exception as exc:
        logger.error("Failed to save OCR audit log: %s", exc)

    try:
        dataset_mgr = get_dataset_manager()
        dataset_mgr.save_sample(
            raw_img_path=saved_path,
            ocr_results=ocr_result,
            human_verified=False,
        )
    except Exception as e:
        logger.error("Failed to log to dataset manager: %s", e)

    return OCRLabResponse(
        raw_text=raw_text,
        corrected_text=final_text,
        reconstructed_text=reconstructed_text,
        final_readable_text=final_text,
        reconstruction_confidence_score=reconstruction_meta.get("reconstruction_confidence_score", 0.0),
        reconstruction_source=reconstruction_meta.get("reconstruction_source", "unknown"),
        detected_domain=reconstruction_meta.get("detected_domain", "unknown"),
        reconstruction_mode=reconstruction_meta.get("reconstruction_mode", reconstruction_mode),
        detected_language=reconstruction_meta.get("detected_language", "unknown"),
        detected_dialect=reconstruction_meta.get("detected_dialect", "unknown"),
        dialect_detection=reconstruction_meta.get("dialect_detection", {}),
        routing_decision=reconstruction_meta.get("routing_decision", "unknown"),
        normalized_dialect_text=reconstruction_meta.get("normalized_dialect_text", ""),
        normalized_regional_text=reconstruction_meta.get("normalized_regional_text", ""),
        translated_text=reconstruction_meta.get("translated_text", ""),
        translation_output=reconstruction_meta.get("translation_output", ""),
        structured_english_complaint=reconstruction_meta.get("structured_english_complaint", final_text),
        entity_preservation_audit=reconstruction_meta.get("entity_preservation_audit", {}),
        language_route_confidence=_safe_float(reconstruction_meta.get("language_route_confidence", 0.0)),
        language_warnings=reconstruction_meta.get("language_warnings", []),
        stage_confidence=reconstruction_meta.get("stage_confidence", {}),
        benchmarks=reconstruction_meta.get("benchmarks", {}),
        raw_confidence=raw_confidence,
        corrected_confidence=corrected_confidence,
        extracted_entities=corrected_entities,
        was_rejected=was_rejected,
        rejection_reason=rejection_reason,
        uncertain_regions=ocr_result.get("uncertain_regions", []),
        officer_review_regions=ocr_result.get("officer_review_regions", []),
        line_results=ocr_result.get("line_results", []),
        preprocessing_metadata=pre_meta,
        processing_time=total_pipeline_runtime_ms,
        file_id=saved_path.name,
        correction_status=reconstruction_meta.get("reconstruction_status", "unknown"),
        timeout_detected=bool(reconstruction_meta.get("timeout_detected", False)),
        fallback_used=bool(reconstruction_meta.get("fallback_used", False)),
        correction_model=reconstruction_meta.get("reconstruction_model", get_active_correction_model()),
        active_ocr_engine=active_engine,
        guardrail_status=reconstruction_meta.get("guardrail_status", "not_checked"),
        lightweight_mode=lightweight_mode,
        lightweight_reason=lightweight_reason,
        performance_metrics=performance_metrics,
        stage_timings=profiler.timings,
    )
