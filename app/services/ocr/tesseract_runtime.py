"""Tesseract path resolution and probes (no Paddle/pipeline imports — avoids circular imports)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def _normalize_tesseract_cmd(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = os.path.expandvars(os.path.expanduser(cleaned))
    return cleaned or None


def resolve_tesseract_cmd() -> str | None:
    """
    Resolve the Tesseract executable: TESSERACT_CMD, PATH, then common Windows paths.
    """
    env_cmd = _normalize_tesseract_cmd(os.getenv("TESSERACT_CMD"))
    if env_cmd:
        p = Path(env_cmd)
        if p.is_file():
            return str(p.resolve())
        logger.warning(
            "TESSERACT_CMD is set but not a file (skipping): %s",
            env_cmd,
        )

    for name in ("tesseract.exe", "tesseract"):
        which = shutil.which(name)
        if which and Path(which).is_file():
            return str(Path(which).resolve())

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None


def get_tesseract_runtime_diagnostics() -> Dict[str, Any]:
    env_raw = os.getenv("TESSERACT_CMD")
    resolved = resolve_tesseract_cmd()
    version = None
    err = None
    if pytesseract is not None and Image is not None:
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
        try:
            version = str(pytesseract.get_tesseract_version())
        except Exception as exc:
            err = str(exc)
    return {
        "tesseract_cmd_env": env_raw,
        "tesseract_cmd_resolved": resolved,
        "tesseract_version": version,
        "pytesseract_installed": pytesseract is not None,
        "pillow_installed": Image is not None,
        "probe_error": err,
    }


def is_tesseract_available() -> bool:
    if pytesseract is None or Image is None:
        return False
    resolved = resolve_tesseract_cmd()
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception as exc:
        logger.warning("Tesseract probe failed: %s", exc)
        return False
