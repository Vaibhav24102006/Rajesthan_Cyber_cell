import logging
import os
import re
import numbers
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.services.ocr.paddle_engine import paddle_engine
from app.services.ocr.tesseract_runtime import (
    get_tesseract_runtime_diagnostics,
    is_tesseract_available,
    resolve_tesseract_cmd,
)

_resolve_tesseract_cmd = resolve_tesseract_cmd

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


logger = logging.getLogger(__name__)


class OCRRuntimeError(RuntimeError):
    """Raised when document OCR cannot run because backends are missing or all attempts failed."""


def _flatten_numeric_values(value: Any) -> List[float]:
    values: List[float] = []
    if isinstance(value, numbers.Number):
        values.append(float(value))
        return values
    if isinstance(value, (list, tuple)):
        for item in value:
            values.extend(_flatten_numeric_values(item))
    return values


def _normalize_confidence(value: Any, context: str) -> float:
    if isinstance(value, bool):
        logger.warning("Boolean confidence in %s; coercing to float: %s", context, value)
        return float(value)
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, (list, tuple)):
        flat = _flatten_numeric_values(value)
        if not flat:
            logger.warning("Non-numeric confidence payload in %s: type=%s value=%r", context, type(value).__name__, value)
            return 0.0
        avg = sum(flat) / len(flat)
        logger.info(
            "Normalized list confidence in %s: entries=%s avg=%.4f",
            context,
            len(flat),
            avg,
        )
        return float(avg)
    logger.warning("Unexpected confidence type in %s: type=%s value=%r", context, type(value).__name__, value)
    return 0.0


def _normalize_ocr_output(result: Dict[str, Any], page_index: int, source: str) -> Dict[str, Any]:
    raw_text = str(result.get("raw_text") or result.get("text") or "")
    confidence = _normalize_confidence(result.get("confidence", result.get("ocr_confidence", 0.0)), f"{source}.page_{page_index}.confidence")
    line_results = result.get("line_results")
    if not isinstance(line_results, list):
        logger.warning(
            "line_results is not a list in %s page=%s type=%s",
            source,
            page_index,
            type(line_results).__name__,
        )
        line_results = []
    uncertain_regions = result.get("uncertain_regions")
    if not isinstance(uncertain_regions, list):
        logger.warning(
            "uncertain_regions is not a list in %s page=%s type=%s",
            source,
            page_index,
            type(uncertain_regions).__name__,
        )
        uncertain_regions = []
    processing_time = _normalize_confidence(result.get("processing_time_ms", 0.0), f"{source}.page_{page_index}.processing_time_ms")

    return {
        "text": raw_text,
        "raw_text": raw_text,
        "confidence": confidence,
        "ocr_confidence": confidence,
        "boxes": line_results,
        "line_results": line_results,
        "uncertain_regions": uncertain_regions,
        "metadata": {
            "source_engine": source,
            "page_index": page_index,
            "raw_confidence_type": type(result.get("confidence", result.get("ocr_confidence", 0.0))).__name__,
        },
        "preprocessing_metadata": result.get("preprocessing_metadata", {}),
        "processing_time_ms": processing_time,
    }


def _ensure_ocr_runtime() -> None:
    missing: List[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("pillow")
    if fitz is None:
        missing.append("pymupdf")
    if missing:
        raise RuntimeError(f"Missing OCR dependencies: {', '.join(missing)}")


def is_paddle_available(load_model: bool = False) -> bool:
    """True if PaddleOCR is usable; health checks should not cold-load the model."""
    if getattr(paddle_engine, "_load_permanently_failed", False):
        return False
    if paddle_engine.is_loaded and paddle_engine.ocr is not None:
        return True
    if not load_model:
        return not getattr(paddle_engine, "_load_permanently_failed", False)
    try:
        paddle_engine._lazy_load()
        return bool(paddle_engine.is_loaded and paddle_engine.ocr is not None)
    except Exception as exc:
        logger.warning("PaddleOCR unavailable: %s", exc)
        return False


def any_ocr_engine_available(load_model: bool = False) -> bool:
    return is_paddle_available(load_model=load_model) or is_tesseract_available()


def assert_document_ocr_runtime() -> None:
    """
    Hard gate for document OCR: at least one of PaddleOCR or Tesseract must work.
    """
    if any_ocr_engine_available(load_model=True):
        return
    diag = get_tesseract_runtime_diagnostics()
    raise OCRRuntimeError(
        "OCR runtime misconfigured: neither PaddleOCR nor Tesseract is usable. "
        "Install paddlepaddle+paddleocr (Python 3.11), install Tesseract OCR, and set "
        "TESSERACT_CMD to tesseract.exe if it is not on PATH. "
        f"Tesseract diagnostics: {diag}"
    )


def get_documents_dir() -> Path:
    configured = os.getenv("DOCUMENTS_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        # repo/backend/storage/documents
        path = Path(__file__).resolve().parents[2] / "storage" / "documents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_ocr_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # Adaptive thresholding works better on scans/photos
    thresh = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return thresh


def _pdf_to_images(pdf_path: Path) -> List[np.ndarray]:
    doc = fitz.open(str(pdf_path))
    pages: List[np.ndarray] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        pages.append(bgr)
    return pages


def _image_file_to_bgr(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return img


def _run_tesseract_ocr(img: np.ndarray) -> Tuple[str, List[float]]:
    if pytesseract is None:
        raise RuntimeError("pytesseract not installed for fallback OCR")
    if Image is None:
        raise RuntimeError("Pillow not installed for fallback OCR")

    resolved = resolve_tesseract_cmd()
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    try:
        text = pytesseract.image_to_string(pil)
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR executable not found. Install Tesseract or set TESSERACT_CMD."
        ) from exc

    confs: List[float] = []
    try:
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
        for raw in data.get("conf", []):
            try:
                v = float(raw)
                if v >= 0:
                    confs.append(v / 100.0)
            except Exception:
                continue
    except Exception:
        pass
    return text, confs


def extract_text_from_document(file_path: Path) -> Dict[str, Any]:
    _ensure_ocr_runtime()
    assert_document_ocr_runtime()

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages = _pdf_to_images(file_path)
    elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        pages = [_image_file_to_bgr(file_path)]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # For debugging, we can define a debug directory based on the file path
    debug_dir = file_path.parent / "debug"
    
    from app.services.ocr.pipeline import extract_text_from_document_pipeline
    
    combined_text = []
    combined_conf = []
    uncertain_regions = []
    line_results = []
    total_time = 0.0
    request_engine_usage: Dict[str, int] = {}
    preprocessing_metadata: Dict[str, Any] = {}
    
    for i, page_img in enumerate(pages):
        page_id = f"{file_path.stem}_p{i}"
        try:
            result = extract_text_from_document_pipeline(page_img, file_id=page_id, debug_dir=debug_dir, fallback_threshold=0.6)
            result = _normalize_ocr_output(result, i, source="ensemble")
        except Exception as e:
            logger.warning(f"Ensemble OCR failed for page {i+1}, using Tesseract fallback: {e}")
            try:
                text, conf = _run_tesseract_ocr(page_img)
                result = {
                    "raw_text": text,
                    "confidence": conf,
                    "uncertain_regions": [],
                    "line_results": [],
                    "processing_time_ms": 0.0,
                }
                result = _normalize_ocr_output(result, i, source="tesseract_fallback")
            except Exception as fallback_error:
                logger.error("Tesseract fallback also failed: %s", fallback_error)
                raise OCRRuntimeError(
                    "OCR failed for this page: ensemble pipeline and Tesseract fallback both failed. "
                    "Fix PaddleOCR/paddlepaddle installation and Tesseract (TESSERACT_CMD / PATH)."
                ) from fallback_error
        
        if result["raw_text"]:
            combined_text.append(result["raw_text"])
        combined_conf.append(result["confidence"])
        uncertain_regions.extend(result["uncertain_regions"])
        line_results.extend(result["line_results"])
        total_time += result["processing_time_ms"]
        preprocessing_metadata = result.get("preprocessing_metadata", preprocessing_metadata)
        metadata = result.get("metadata", {})
        source_engine = metadata.get("source_engine", "unknown")
        page_engine_usage = preprocessing_metadata.get("engine_usage", {}) if isinstance(preprocessing_metadata, dict) else {}
        if isinstance(page_engine_usage, dict) and page_engine_usage:
            for engine_name, count in page_engine_usage.items():
                try:
                    request_engine_usage[str(engine_name)] = request_engine_usage.get(str(engine_name), 0) + int(count)
                except Exception:
                    continue
        else:
            request_engine_usage[source_engine] = request_engine_usage.get(source_engine, 0) + 1
        logger.info(
            "OCR page aggregation page=%s confidence=%.4f conf_type=%s line_results=%s uncertain_regions=%s",
            i,
            result["confidence"],
            type(result["confidence"]).__name__,
            len(result["line_results"]),
            len(result["uncertain_regions"]),
        )

    logger.info(
        "Final OCR aggregation inputs confidences=%r types=%r",
        combined_conf,
        [type(v).__name__ for v in combined_conf],
    )
    active_engine = max(request_engine_usage, key=request_engine_usage.get) if request_engine_usage else "unknown"
    logger.info(
        "OCR request runtime verification: Tesseract initialized=%s, active OCR engine=%s, usage=%s",
        is_tesseract_available(),
        active_engine,
        request_engine_usage,
    )
    avg_conf = sum(combined_conf) / len(combined_conf) if combined_conf else 0.0
    
    return {
        "text": "\n".join(combined_text),
        "raw_text": "\n".join(combined_text),
        "ocr_confidence": round(max(0.0, min(avg_conf, 1.0)), 4),
        "confidence": round(max(0.0, min(avg_conf, 1.0)), 4),
        "ocr_engine": "ensemble",
        "uncertain_regions": uncertain_regions,
        "line_results": line_results,
        "pages": len(pages),
        "processing_time_ms": total_time,
        "preprocessing_metadata": {
            **(preprocessing_metadata if isinstance(preprocessing_metadata, dict) else {}),
            "engine_usage": request_engine_usage,
            "active_ocr_engine": active_engine,
        },
    }
