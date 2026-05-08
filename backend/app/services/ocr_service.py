import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

_PADDLE_OCR = None


def _resolve_tesseract_cmd() -> str | None:
    # Allow explicit override from environment first.
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    # Common Windows install locations.
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


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


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    try:
        from paddleocr import PaddleOCR  # type: ignore

        _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en")
        return _PADDLE_OCR
    except Exception as exc:
        logger.warning("PaddleOCR unavailable, fallback to Tesseract only: %s", exc)
        return None


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


def _run_paddle_ocr(img: np.ndarray) -> Tuple[str, List[float]]:
    ocr = _get_paddle_ocr()
    if ocr is None:
        raise RuntimeError("PaddleOCR unavailable")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(rgb, cls=True)

    lines: List[str] = []
    confs: List[float] = []
    for block in result or []:
        if not block:
            continue
        for entry in block:
            try:
                txt = str(entry[1][0]).strip()
                conf = float(entry[1][1])
            except Exception:
                continue
            if txt:
                lines.append(txt)
                confs.append(conf)
    return "\n".join(lines), confs


def _run_tesseract_ocr(img: np.ndarray) -> Tuple[str, List[float]]:
    if pytesseract is None:
        raise RuntimeError("pytesseract not installed for fallback OCR")
    if Image is None:
        raise RuntimeError("Pillow not installed for fallback OCR")

    resolved = _resolve_tesseract_cmd()
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

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages = _pdf_to_images(file_path)
    elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        pages = [_image_file_to_bgr(file_path)]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    all_text_chunks: List[str] = []
    all_conf: List[float] = []
    engine_used = "paddleocr"
    paddle_failures = 0

    for page_img in pages:
        processed = preprocess_image(page_img)
        # OCR engines expect 3 channel for best compatibility
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        text = ""
        confs: List[float] = []
        try:
            text, confs = _run_paddle_ocr(processed_bgr)
        except Exception:
            paddle_failures += 1

        if not text.strip():
            # fallback to local Tesseract
            engine_used = "tesseract_fallback"
            text, confs = _run_tesseract_ocr(processed_bgr)

        if text.strip():
            all_text_chunks.append(text)
        if confs:
            all_conf.extend(confs)

    full_text = clean_ocr_text("\n".join(all_text_chunks))
    if not full_text:
        raise ValueError("OCR produced empty text")

    if paddle_failures == len(pages):
        engine_used = "tesseract_fallback"

    avg_conf = float(sum(all_conf) / len(all_conf)) if all_conf else 0.0
    return {
        "text": full_text,
        "ocr_confidence": round(max(0.0, min(avg_conf, 1.0)), 4),
        "ocr_engine": engine_used,
        "pages": len(pages),
    }

