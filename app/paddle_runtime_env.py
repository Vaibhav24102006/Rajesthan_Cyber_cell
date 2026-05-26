"""
Point Paddle / PaddleX caches at backend/storage (avoids locked ~/.cache or ~/.paddlex).

Call `ensure_paddle_runtime_env()` before importing paddleocr or paddle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_done = False


def ensure_paddle_runtime_env(backend_dir: Path) -> None:
    """Idempotent: set PADDLE_PDX_CACHE_HOME, XDG_CACHE_HOME, PADDLE_EXTENSION_DIR under storage."""
    global _done
    if _done:
        return
    root = (backend_dir / "storage" / "paddle_runtime").resolve()
    paddlex = root / "paddlex"
    xdg_parent = root / "xdg_cache_home"
    extension = root / "paddle_extension"
    try:
        for p in (root, paddlex, xdg_parent, extension):
            p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create Paddle runtime directories under %s: %s", root, exc)
        return

    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paddlex))
    # hub._get_paddle_home joins XDG_CACHE_HOME with "paddle"
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_parent))
    os.environ.setdefault("PADDLE_EXTENSION_DIR", str(extension))
    
    # Disable PIR API to fix ConvertPirAttribute2RuntimeAttribute bugs on Windows (Paddle 3.0b+)
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    # paddle.dataset.common uses expanduser("~")/.cache/paddle (Windows: often USERPROFILE).
    cache_probe = Path.home() / ".cache" / "paddle"
    need_profile_redirect = False
    try:
        cache_probe.mkdir(parents=True, exist_ok=True)
        probe_file = cache_probe / ".jaipur_cybercell_write_test"
        probe_file.write_text("ok", encoding="ascii")
        probe_file.unlink(missing_ok=True)  # py3.8+ missing_ok
    except OSError:
        need_profile_redirect = True

    if need_profile_redirect:
        fake_home = root / "fake_user_home"
        fake_home.mkdir(parents=True, exist_ok=True)
        fh = str(fake_home.resolve())
        logger.warning(
            "User cache at %s is not writable; redirecting HOME/USERPROFILE to %s for this process.",
            cache_probe,
            fh,
        )
        os.environ["HOME"] = fh
        os.environ["USERPROFILE"] = fh

    _done = True
