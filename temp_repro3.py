import sys
import os
import time
import faulthandler
from pathlib import Path
from app.paddle_runtime_env import ensure_paddle_runtime_env
fal = faulthandler.enable()
print('START', flush=True)
backend_dir = Path('.').resolve()
ensure_paddle_runtime_env(backend_dir)
print('ENV', os.environ.get('PADDLE_PDX_CACHE_HOME'), os.environ.get('XDG_CACHE_HOME'), flush=True)
try:
    from paddleocr import PaddleOCR
    print('imported paddleocr', flush=True)
    ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False)
    print('loaded', type(ocr), flush=True)
except Exception as e:
    print('EXCEPTION', repr(e), file=sys.stderr, flush=True)
    sys.exit(1)
print('AFTER', flush=True)
time.sleep(1)
