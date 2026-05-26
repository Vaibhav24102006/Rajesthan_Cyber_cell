import logging
import os
import time
from pathlib import Path
from typing import Tuple

from app.paddle_runtime_env import ensure_paddle_runtime_env

_backend_dir = Path(__file__).resolve().parents[3]
ensure_paddle_runtime_env(_backend_dir)

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class PaddleEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PaddleEngine, cls).__new__(cls)
            cls._instance.ocr = None
            cls._instance.is_loaded = False
            cls._instance.current_lang = None
            cls._instance._load_permanently_failed = False
            cls._instance.last_used = 0
            cls._instance.auto_unload_enabled = True
            cls._instance.unload_timeout = 300
        return cls._instance

    def free_memory(self):
        if self.is_loaded:
            logger.info("Freeing PaddleOCR memory...")
            self.ocr = None
            self.is_loaded = False
            self._load_permanently_failed = False
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _lazy_load(self, lang: str = None):
        lang = (lang or os.getenv("OCR_PADDLE_LANG", "en")).strip() or "en"
        if getattr(self, "_load_permanently_failed", False):
            raise RuntimeError("PaddleOCR initialization failed earlier in this process")

        if self.auto_unload_enabled and self.is_loaded and (time.time() - self.last_used > self.unload_timeout):
            self.free_memory()

        if self.is_loaded and self.current_lang != lang:
            logger.info("Switching PaddleOCR language model from %s to %s", self.current_lang, lang)
            self.free_memory()

        if not self.is_loaded:
            logger.info("Lazy loading PaddleOCR model (lang=%s)...", lang)
            start = time.time()
            try:
                from paddleocr import PaddleOCR

                det_model = os.getenv("OCR_PADDLE_DET_MODEL", "").strip() or None
                rec_model = os.getenv("OCR_PADDLE_REC_MODEL", "").strip() or None
                self.ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang=lang,
                    text_detection_model_name=det_model,
                    text_recognition_model_name=rec_model,
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
                self.is_loaded = True
                self.current_lang = lang
                logger.info("PaddleOCR model loaded in %.2fs", time.time() - start)
            except Exception as e:
                self._load_permanently_failed = True
                self.ocr = None
                self.is_loaded = False
                logger.error("Failed to load PaddleOCR: %s", e)
                raise

    def run_ocr(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Runs PaddleOCR on a single line image.
        """
        self._lazy_load()
        
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = self.ocr.ocr(rgb)
            
            lines = []
            confs = []
            for block in result or []:
                if not block:
                    continue
                for entry in block:
                    try:
                        txt = str(entry[1][0]).strip()
                        conf = float(entry[1][1])
                        if txt:
                            lines.append(txt)
                            confs.append(conf)
                    except Exception:
                        continue
            
            text = " ".join(lines)
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            
            self.last_used = time.time()
            return text.strip(), avg_conf
            
        except Exception as e:
            logger.error(f"PaddleOCR inference failed: {e}")
            self.last_used = time.time()
            return "", 0.0

# Singleton instance
paddle_engine = PaddleEngine()
