import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

from app.paddle_runtime_env import ensure_paddle_runtime_env

ensure_paddle_runtime_env(BASE_DIR)

from app.database.db import Base, engine
from app.routes.extract import router as extract_router
from app.routes.analytics import router as analytics_router
from app.routes.query import router as query_router
from app.routes.similarity import router as similarity_router
from app.routes.audit import router as audit_router
from app.routes.ocr_lab import router as ocr_lab_router
from app.services.indus_reconstruction_service import INDUS_API_URL, INDUS_HEALTH_URL, is_indus_reachable
from app.services.ocr_service import (
    get_tesseract_runtime_diagnostics,
    is_paddle_available,
    is_tesseract_available,
    any_ocr_engine_available,
)
from app.services.ocr_correction import get_active_correction_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jaipur Cyber Cell API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router)
app.include_router(analytics_router)
app.include_router(query_router)
app.include_router(similarity_router)
app.include_router(audit_router)
app.include_router(ocr_lab_router, prefix="/ocr-lab", tags=["OCR Lab"])
    get_tesseract_runtime_diagnostics,
    is_paddle_available,
    is_tesseract_available,
    any_ocr_engine_available,
)
from app.services.ocr_correction import get_active_correction_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jaipur Cyber Cell API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router)
app.include_router(analytics_router)
app.include_router(query_router)
app.include_router(similarity_router)
app.include_router(audit_router)
app.include_router(ocr_lab_router, prefix="/ocr-lab", tags=["OCR Lab"])


debug_dir = BASE_DIR / "storage" / "documents" / "debug"
debug_dir.mkdir(parents=True, exist_ok=True)
app.mount("/debug", StaticFiles(directory=str(debug_dir)), name="debug")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized and API started")
    logger.info(
        "OCR startup config | active_ocr_engine=%s | correction_model=%s | pipeline_mode=%s",
        "paddle_primary_tesseract_fallback",
        get_active_correction_model(),
        "low_memory_cpu",
    )
    
    # ────────────────────────────────────────────────────────────────
    # HARDENED OCR STARTUP INTEGRITY RUNTIME CHECK (FAIL-FAST)
    # ────────────────────────────────────────────────────────────────
    logger.info("==================================================")
    logger.info("Running Hardened OCR Startup Verification Hook...")
    logger.info("==================================================")
    
    try:
        # 1. Verify all 10 Preprocessing Profiles Compile
        import numpy as np
        from app.services.ocr.preprocessing import preprocess_for_ocr
        dummy_canvas = np.ones((128, 128, 3), dtype=np.uint8) * 255
        profiles = [
            "adaptive", "otsu", "denoise-heavy", "contrast-heavy", 
            "sharpen-heavy", "handwriting-clahe", "handwriting-blue-ink", 
            "handwriting-fast", "handwriting-devanagari", "auto"
        ]
        for p in profiles:
            logger.info("Pre-compiling and verifying profile: %s", p)
            _ = preprocess_for_ocr(dummy_canvas, profile=p)
        logger.info("[+] All 10 image preprocessing profiles compiled successfully.")
        
        # 2. Verify PaddleOCR Active Inference (preload Hindi hi, run CPU inference)
        from app.services.ocr.paddle_engine import paddle_engine
        logger.info("Initializing and preloading PaddleOCR with Hindi (hi) language model...")
        paddle_engine._lazy_load(lang="hi")
        
        logger.info("Executing active OCR test inference on dummy canvas...")
        test_text, test_conf = paddle_engine.run_ocr(dummy_canvas)
        logger.info("[+] Active OCR inference test completed successfully. Result: '%s' (conf=%.4f)", test_text, test_conf)
        
    except Exception as exc:
        logger.critical("[-] CRITICAL OCR RUNTIME FAIL-FAST INITIALIZATION FAILURE!")
        logger.critical("Error Details: %s", exc, exc_info=True)
        logger.critical("OCR environment is unstable or improperly configured. Halting startup immediately.")
        sys.exit(1)
        
    logger.info("==================================================")
    logger.info("OCR Runtime verification successful! API is READY.")
    logger.info("==================================================")

    preload_paddle = os.getenv("OCR_PRELOAD_PADDLE_ON_STARTUP", "").strip().lower() in {"1", "true", "yes", "on"}
    paddle_ok = is_paddle_available(load_model=preload_paddle)
    tess_ok = is_tesseract_available()
    tess_diag = get_tesseract_runtime_diagnostics()
    logger.info(
        "OCR runtime verification | PaddleOCR initialized=%s | Tesseract initialized=%s",
        paddle_ok,
        tess_ok,
    )
    logger.info(
        "Tesseract diagnostics | cmd_resolved=%s | version=%s | probe_error=%s",
        tess_diag.get("tesseract_cmd_resolved"),
        tess_diag.get("tesseract_version"),
        tess_diag.get("probe_error"),
    )
    logger.info(
        "Indus runtime verification | api_url=%s health_url=%s reachable=%s",
        INDUS_API_URL or "<unset>",
        INDUS_HEALTH_URL or "<unset>",
        is_indus_reachable(),
    )
    strict = os.getenv("OCR_STRICT_STARTUP", "").strip().lower() in {"1", "true", "yes", "on"}
    if strict and not any_ocr_engine_available(load_model=preload_paddle):
        logger.critical(
            "OCR_STRICT_STARTUP is enabled but no OCR engine is available (Paddle=%s, Tesseract=%s). Exiting.",
            paddle_ok,
            tess_ok,
        )
        sys.exit(1)
    if not any_ocr_engine_available(load_model=preload_paddle):
        logger.error(
            "OCR engines unavailable: document OCR will return errors until PaddleOCR "
            "and/or Tesseract is fixed. Set OCR_STRICT_STARTUP=true to fail fast on boot."
        )
    # Lightweight SQLite schema migration:
        from app.models.complaint import Complaint

        if str(engine.url).startswith("sqlite"):
            insp = inspect(engine)
            
            # Check OCR Audit Logs table
            from app.models.audit_log import OCRAuditLog
            if not insp.has_table(OCRAuditLog.__tablename__):
                OCRAuditLog.__table__.create(engine)

            if insp.has_table(Complaint.__tablename__):
                existing_cols = {c["name"] for c in insp.get_columns(Complaint.__tablename__)}
                with engine.begin() as conn:
                    for col in Complaint.__table__.columns:
                        if col.name in existing_cols:
                            continue
                        col_type = "TEXT"
                        if col.type.__class__.__name__ == "Integer":
                            col_type = "INTEGER"
                        elif col.type.__class__.__name__ == "Float":
                            col_type = "REAL"
                        elif col.type.__class__.__name__ == "Boolean":
                            col_type = "INTEGER"
                        elif col.type.__class__.__name__ == "DateTime":
                            col_type = "DATETIME"
                        elif col.type.__class__.__name__ in {"String", "Text"}:
                            col_type = "TEXT"
                        elif col.type.__class__.__name__ == "JSON":
                            col_type = "TEXT"
                        conn.execute(
                            text(f"ALTER TABLE {Complaint.__tablename__} ADD COLUMN {col.name} {col_type}")
                        )
            logger.info("SQLite schema check completed")
    except Exception as exc:
        logger.warning("SQLite migration skipped/failed: %s", exc)


@app.get("/health")
def health_check():
    return {"status": "ok"}
