import logging
import numbers
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from app.services.ocr.layout_analysis import analyze_layout
from app.services.ocr.paddle_engine import paddle_engine
from app.services.ocr.preprocessing import annotate_image, preprocess_for_ocr
from app.services.ocr.script_detection import classify_script_visually
from app.services.ocr.tesseract_runtime import resolve_tesseract_cmd

logger = logging.getLogger(__name__)

DEVANAGARI_UNICODE_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_UNICODE_RANGE = re.compile(r"[A-Za-z]")


def _coerce_confidence(value: Any, context: str) -> float:
    if isinstance(value, bool):
        logger.warning("Boolean confidence in %s; coercing to float: %s", context, value)
        return float(value)
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for item in value:
            if isinstance(item, numbers.Number):
                flattened.append(float(item))
        if flattened:
            avg = sum(flattened) / len(flattened)
            logger.info("Coerced list confidence in %s to avg=%.4f (n=%s)", context, avg, len(flattened))
            return float(avg)
        logger.warning("Non-numeric list confidence in %s: %r", context, value)
        return 0.0
    logger.warning("Unexpected confidence type in %s: type=%s value=%r", context, type(value).__name__, value)
    return 0.0


def _run_tesseract_ocr(line_img_bgr: np.ndarray) -> tuple[str, float]:
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Tesseract fallback unavailable (missing pytesseract/Pillow).") from exc

    resolved = resolve_tesseract_cmd()
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved

    pil = Image.fromarray(cv2.cvtColor(line_img_bgr, cv2.COLOR_BGR2RGB))
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
                value = float(raw)
                if value >= 0:
                    confs.append(value / 100.0)
            except Exception:
                continue
    except Exception:
        pass

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text or "", avg_conf


def _box_to_points(box: Any, scale_factor: float) -> List[List[int]]:
    arr = np.asarray(box)
    if arr.shape == (4,):
        x1, y1, x2, y2 = [int(v / scale_factor) for v in arr.tolist()]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    points = arr.reshape(-1, 2)[:4]
    return [[int(p[0] / scale_factor), int(p[1] / scale_factor)] for p in points]


def _extract_paddle_blocks(full_page_result: Any, scale_factor: float) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for page in full_page_result or []:
        if not page:
            continue
        if isinstance(page, dict):
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("dt_polys")
            if polys is None:
                polys = []
            boxes = page.get("rec_boxes")
            if boxes is None:
                boxes = []
            for idx, text in enumerate(texts):
                txt = str(text).strip()
                if not txt:
                    continue
                score = _coerce_confidence(scores[idx] if idx < len(scores) else 0.0, f"paddle_v3.score_{idx}")
                raw_box = polys[idx] if idx < len(polys) else (boxes[idx] if idx < len(boxes) else None)
                if raw_box is None:
                    continue
                try:
                    blocks.append({"box": _box_to_points(raw_box, scale_factor), "text": txt, "conf": score})
                except Exception as exc:
                    logger.warning("Unable to parse PaddleOCR v3 box %s: %s", idx, exc)
            continue

        for entry in page:
            try:
                orig_box = entry[0]
                box = _box_to_points(orig_box, scale_factor)
                txt = str(entry[1][0]).strip()
                conf = float(entry[1][1])
                if txt:
                    blocks.append({"box": box, "text": txt, "conf": conf})
            except Exception:
                continue
    return blocks


def _sort_blocks_reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def center(block: Dict[str, Any]) -> tuple[float, float]:
        pts = block.get("box") or []
        if not pts:
            return (0.0, 0.0)
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    if not blocks:
        return []

    heights = []
    for block in blocks:
        pts = block.get("box") or []
        if pts:
            ys = [float(p[1]) for p in pts]
            heights.append(max(ys) - min(ys))
    row_tolerance = max(12.0, (sum(heights) / len(heights)) * 0.65) if heights else 18.0

    rows: List[List[Dict[str, Any]]] = []
    for block in sorted(blocks, key=lambda item: center(item)[0]):
        cy, _ = center(block)
        if not rows:
            rows.append([block])
            continue
        row_cy = sum(center(item)[0] for item in rows[-1]) / len(rows[-1])
        if abs(cy - row_cy) <= row_tolerance:
            rows[-1].append(block)
        else:
            rows.append([block])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: center(item)[1]))
    return ordered


def _calculate_devanagari_recovery_score(text: str, ocr_confidence: float) -> Dict:
    if not text.strip():
        return {"devanagari_recovery_score": 0.0, "hindi_token_readability": 0.0,
                "devanagari_ratio": 0.0, "devanagari_confidence": 0.0}
    total_chars = len(text.replace(" ", ""))
    dev_chars = len(DEVANAGARI_UNICODE_RANGE.findall(text))
    latin_chars = len(LATIN_UNICODE_RANGE.findall(text))
    dev_ratio = dev_chars / max(1, total_chars)
    latin_ratio = latin_chars / max(1, total_chars)

    tokens = re.findall(r"[\w\u0900-\u097F]+", text)
    if not tokens:
        return {"devanagari_recovery_score": 0.0, "hindi_token_readability": 0.0,
                "devanagari_ratio": dev_ratio, "devanagari_confidence": 0.0}
    dev_tokens = sum(1 for t in tokens if DEVANAGARI_UNICODE_RANGE.search(t))
    dev_token_ratio = dev_tokens / len(tokens)

    dev_confidence = dev_ratio * 0.5 + dev_token_ratio * 0.3 + ocr_confidence * 0.2
    readability = 0.0
    if dev_tokens > 0:
        dev_token_text = " ".join(t for t in tokens if DEVANAGARI_UNICODE_RANGE.search(t))
        avg_token_len = len(dev_token_text.replace(" ", "")) / max(1, dev_tokens)
        readability = min(1.0, avg_token_len / 6.0)

    return {
        "devanagari_recovery_score": round(dev_confidence, 4),
        "hindi_token_readability": round(readability, 4),
        "devanagari_ratio": round(dev_ratio, 4),
        "latin_ratio": round(latin_ratio, 4),
        "devanagari_token_ratio": round(dev_token_ratio, 4),
        "devanagari_confidence": round(dev_confidence, 4),
    }


def _calculate_weighted_score(text: str, base_conf: float) -> float:
    base_conf = _coerce_confidence(base_conf, "calculate_weighted_score.base_conf")
    if not text:
        return 0.0
    alpha_num_count = len(re.findall(r"[A-Za-z0-9\u0900-\u097F]", text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return 0.0
    lexical_validity = alpha_num_count / total_chars
    return (base_conf * 0.8) + (lexical_validity * 0.2)


def _should_officer_review(devanagari_recovery: Dict, script_info: Dict, ocr_confidence: float) -> Dict:
    reasons = []
    if script_info["script"] in ("devanagari", "mixed"):
        if devanagari_recovery["devanagari_recovery_score"] < 0.15:
            reasons.append("devanagari_recovery_too_low_for_translation")
        if devanagari_recovery["devanagari_ratio"] < 0.05 and script_info["script"] == "devanagari":
            reasons.append("expected_devanagari_but_ocr_produced_latin_garbage")
    if ocr_confidence < 0.60:
        reasons.append("low_ocr_confidence")
    if script_info["confidence"] < 0.35:
        reasons.append("unstable_script_detection")
    return {
        "officer_review_required": len(reasons) > 0,
        "review_reasons": reasons,
    }


def extract_text_from_document_pipeline(
    img_bgr: np.ndarray,
    file_id: str = "doc",
    debug_dir: Path = None,
    fallback_threshold: float = 0.5,
    preprocessing_profile: str = "auto",
    paddle_lang: str = None,
    paddle_max_side: int = None,
) -> Dict[str, Any]:
    overall_start = time.time()
    logger.info("[START] OCR pipeline execution")
    logger.info("Uploaded image dimensions: %s", img_bgr.shape)

    # ──────────────────────────────────────────────────
    # PHASE 0: VISUAL SCRIPT DETECTION (before any OCR)
    # ──────────────────────────────────────────────────
    script_start = time.time()
    script_info = classify_script_visually(img_bgr)
    script_time = round((time.time() - script_start) * 1000, 2)
    logger.info(
        "Visual script detection: script=%s confidence=%.4f (took %.2fms)",
        script_info["script"], script_info["confidence"], script_time
    )

    # ──────────────────────────────────────────────────
    # PHASE 1: SCRIPT-AWARE PREPROCESSING ROUTING
    # ──────────────────────────────────────────────────
    primary_profile = preprocessing_profile or os.getenv("OCR_PREPROCESS_PROFILE", "auto")
    if primary_profile == "auto" and script_info["script"] == "devanagari":
        if script_info["confidence"] > 0.35:
            primary_profile = "handwriting-devanagari"
            logger.info("Script routing: Devanagari detected -> using handwriting-devanagari profile")
    elif primary_profile == "auto" and script_info["script"] == "mixed":
        if script_info["confidence"] > 0.30:
            primary_profile = "handwriting-devanagari"
            logger.info("Script routing: Mixed script detected -> using handwriting-devanagari profile")

    # Resolve paddle language based on detected script
    original_paddle_lang = (paddle_lang or os.getenv("OCR_PADDLE_LANG", "en")).strip() or "en"
    if script_info["script"] == "devanagari" and script_info["confidence"] > 0.30:
        if original_paddle_lang == "en":
            paddle_lang = "hi"
            logger.info("Script routing: Devanagari detected -> switching PaddleOCR lang from 'en' to 'hi'")
        else:
            paddle_lang = original_paddle_lang
    elif script_info["script"] == "mixed" and script_info["confidence"] > 0.25:
        if original_paddle_lang == "en":
            paddle_lang = "hi"
            logger.info("Script routing: Mixed script detected -> switching PaddleOCR lang to 'hi'")
        else:
            paddle_lang = original_paddle_lang
    else:
        paddle_lang = original_paddle_lang

    paddle_max_side = int(paddle_max_side or os.getenv("OCR_PADDLE_MAX_SIDE", "1280"))

    logger.info("[START] image preprocessing (profile=%s, paddle_lang=%s)", primary_profile, paddle_lang)
    prep_result = preprocess_for_ocr(img_bgr, profile=primary_profile, debug_dir=debug_dir, file_id=file_id)
    processed_bgr = prep_result["processed_bgr"]
    processed_thresh = prep_result["processed_thresh"]
    logger.info("[END] image preprocessing. Final resolution: %s", processed_bgr.shape)
    heuristics = prep_result["metadata"].get("heuristics", {})
    actual_profile = prep_result["metadata"]["profile"]

    failure_reasons = set()
    if heuristics.get("skew_angle", 0) > 5.0:
        failure_reasons.add("High Skew Detected")
    if heuristics.get("noise_density", 100) < 50:
        failure_reasons.add("High Noise Corruption")
    if heuristics.get("contrast_score", 100) < 35:
        failure_reasons.add("Low Contrast Document")

    seg_start = time.time()
    logger.info("[START] layout analysis and segmentation")
    line_data = analyze_layout(processed_thresh, processed_bgr, debug_dir=debug_dir, file_id=file_id)
    seg_time = round((time.time() - seg_start) * 1000, 2)
    logger.info("[END] layout analysis and segmentation (found %s regions)", len(line_data))
    prep_result["metadata"]["segmentation_time_ms"] = seg_time

    ocr_start = time.time()
    raw_text_lines = []
    line_results = []
    uncertain_regions = []
    officer_review_regions = []
    skipped_regions = []
    all_confs = []
    engine_used_counts = {"paddle": 0, "tesseract": 0, "skipped": 0}

    # ──────────────────────────────────────────────────
    # PHASE 2: SCRIPT-AWARE OCR
    # ──────────────────────────────────────────────────
    # Load PaddleOCR with the resolved language
    paddle_engine._lazy_load(lang=paddle_lang)

    # HARD LIMIT: Resize image if too large to prevent CPU stall
    max_dim = max(640, paddle_max_side)
    h, w = processed_bgr.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        logger.warning("Image too large (%sx%s). Resizing to %sx%s for PaddleOCR.", w, h, new_w, new_h)
        paddle_input_bgr = cv2.resize(processed_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        paddle_input_bgr = processed_bgr

    logger.info("[START] PaddleOCR inference (lang=%s, resolution=%s)", paddle_lang, paddle_input_bgr.shape)
    paddle_start_time = time.time()
    try:
        rgb = cv2.cvtColor(paddle_input_bgr, cv2.COLOR_BGR2RGB)
        full_page_result = paddle_engine.ocr.ocr(rgb)
    except Exception as e:
        logger.error("PaddleOCR full page inference failed: %s", e)
        full_page_result = []
    logger.info("[END] PaddleOCR inference (took %.2fs)", time.time() - paddle_start_time)

    blocks = _sort_blocks_reading_order(_extract_paddle_blocks(full_page_result, scale_factor))
    logger.info("PaddleOCR parsed text blocks=%s", len(blocks))

    # ──────────────────────────────────────────────────
    # PHASE 3: MIXED-SCRIPT REGION OCR
    # ──────────────────────────────────────────────────
    # If script is mixed, attempt re-OCR of pure-English regions with English lang
    for i, block in enumerate(blocks):
        box = block["box"]
        y_coords = (int(min(p[1] for p in box)), int(max(p[1] for p in box)))
        x_coords = (int(min(p[0] for p in box)), int(max(p[0] for p in box)))

        y1, y2 = max(0, y_coords[0] - 2), min(processed_bgr.shape[0], y_coords[1] + 2)
        x1, x2 = max(0, x_coords[0] - 2), min(processed_bgr.shape[1], x_coords[1] + 2)
        line_img = processed_bgr[y1:y2, x1:x2]

        fallback_start = time.time()

        best_text = block["text"]
        best_conf = block["conf"]
        best_engine = "paddle"
        best_profile = actual_profile
        best_weighted_score = _calculate_weighted_score(best_text, best_conf)

        region_type = "paragraph"

        # If mixed script, try English Paddle re-OCR on blocks that look like Latin text
        if (script_info["script"] == "mixed"
                and best_weighted_score < fallback_threshold
                and line_img.size > 0):
            latin_ratio_in_block = len(LATIN_UNICODE_RANGE.findall(best_text)) / max(1, len(best_text.replace(" ", "")))
            if latin_ratio_in_block > 0.6:
                logger.info("Mixed script routing: re-OCRing block %s with English PaddleOCR", i)
                try:
                    paddle_engine._lazy_load(lang="en")
                    rgb_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
                    en_result = paddle_engine.ocr.ocr(rgb_line)
                    en_blocks = _extract_paddle_blocks(en_result, 1.0)
                    if en_blocks:
                        en_text = " ".join(b["text"] for b in en_blocks)
                        en_conf = float(np.mean([b["conf"] for b in en_blocks])) if en_blocks else 0.0
                        en_score = _calculate_weighted_score(en_text, en_conf)
                        if en_score > best_weighted_score:
                            best_text, best_conf, best_weighted_score, best_engine = en_text, en_conf, en_score, "paddle_en"
                            logger.info("Mixed script: English re-OCR improved block %s (%.4f -> %.4f)", i, best_weighted_score, en_score)
                except Exception as reocr_err:
                    logger.warning("Mixed script re-OCR failed for block %s: %s", i, reocr_err)
                finally:
                    paddle_engine._lazy_load(lang=paddle_lang)

        # Fallback to Tesseract if confidence is low
        if best_weighted_score < fallback_threshold and line_img.size > 0:
            logger.warning("PaddleOCR confidence low (%.2f) for line %s, using Tesseract fallback", best_weighted_score, i)
            try:
                tf_text, tf_conf = _run_tesseract_ocr(line_img)
                tf_conf = _coerce_confidence(tf_conf, f"line_{i}.tesseract_fallback_conf")
                tf_score = _calculate_weighted_score(tf_text, tf_conf)
                if tf_score > best_weighted_score:
                    best_text, best_conf, best_weighted_score, best_engine = tf_text, tf_conf, tf_score, "tesseract"
            except Exception as fallback_error:
                logger.error("Tesseract fallback failed for line %s: %s", i, fallback_error)

        fallback_duration = time.time() - fallback_start
        if fallback_duration > 5.0:
            logger.warning("Fallback OCR took %.2fs for line %s, may cause delays", fallback_duration, i)

        if best_text.strip():
            raw_text_lines.append(best_text.strip())
            all_confs.append(best_weighted_score)
            engine_used_counts.setdefault(best_engine, 0)
            engine_used_counts[best_engine] += 1
            res_item = {
                "line": i,
                "text": best_text.strip(),
                "confidence": best_weighted_score,
                "engine": best_engine,
                "profile": best_profile,
                "retries": 0,
                "y_coords": y_coords,
                "x_coords": x_coords,
                "box": box,
                "region_type": region_type,
            }
            line_results.append(res_item)
            if best_weighted_score < fallback_threshold:
                uncertain_regions.append(res_item)
            if best_weighted_score < 0.75:
                officer_review_regions.append(res_item)

    # ──────────────────────────────────────────────────
    # PHASE 4: SEGMENTATION FALLBACK (if Paddle found nothing)
    # ──────────────────────────────────────────────────
    if not blocks:
        logger.warning("PaddleOCR full page detection found NO text! Falling back to segmentation + Tesseract.")
        failure_reasons.add("Paddle Detection Failure")
        for i, data in enumerate(line_data):
            line_img = data["image"]
            y_coords = data["coords"]
            try:
                best_text, best_conf = _run_tesseract_ocr(line_img)
                best_weighted_score = _calculate_weighted_score(best_text, best_conf)
                if best_text.strip():
                    raw_text_lines.append(best_text.strip())
                    all_confs.append(best_weighted_score)
                    engine_used_counts["tesseract"] += 1
                    line_results.append({
                        "line": i,
                        "text": best_text.strip(),
                        "confidence": best_weighted_score,
                        "engine": "tesseract",
                        "profile": actual_profile,
                        "retries": 0,
                        "y_coords": y_coords,
                        "x_coords": (0, line_img.shape[1]),
                        "box": [[0, y_coords[0]], [line_img.shape[1], y_coords[0]], [line_img.shape[1], y_coords[1]], [0, y_coords[1]]],
                        "region_type": "paragraph",
                    })
                    if best_weighted_score < 0.75:
                        officer_review_regions.append(line_results[-1])
            except Exception as fallback_error:
                logger.error("Segmentation Tesseract fallback failed for line %s: %s", i, fallback_error)

    annotated_img_path = annotate_image(processed_bgr, line_results, debug_dir, file_id)

    ocr_time = round((time.time() - ocr_start) * 1000, 2)
    raw_text = "\n".join(raw_text_lines)

    # ──────────────────────────────────────────────────
    # PHASE 5: DEVANAGARI RECOVERY ASSESSMENT
    # ──────────────────────────────────────────────────
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    devanagari_recovery = _calculate_devanagari_recovery_score(raw_text, avg_conf)

    # ──────────────────────────────────────────────────
    # PHASE 6: OFFICER-REVIEW SAFETY GATE
    # ──────────────────────────────────────────────────
    review_gate = _should_officer_review(devanagari_recovery, script_info, avg_conf)
    translation_blocked = False
    if review_gate["officer_review_required"]:
        if script_info["script"] in ("devanagari", "mixed") and devanagari_recovery["devanagari_recovery_score"] < 0.15:
            translation_blocked = True
            logger.warning(
                "TRANSLATION BLOCKED: Devanagari recovery score %.4f is below threshold. "
                "Returning officer-review warning.",
                devanagari_recovery["devanagari_recovery_score"]
            )

    prep_result["metadata"]["ocr_inference_time_ms"] = ocr_time
    prep_result["metadata"]["engine_usage"] = engine_used_counts
    prep_result["metadata"]["failure_reasons"] = list(failure_reasons)
    prep_result["metadata"]["skipped_regions"] = skipped_regions
    prep_result["metadata"]["paddle_lang"] = paddle_lang
    prep_result["metadata"]["paddle_max_side"] = max_dim
    prep_result["metadata"]["officer_review_required"] = bool(officer_review_regions) or review_gate["officer_review_required"]
    prep_result["metadata"]["officer_review_region_count"] = len(officer_review_regions)

    # Attach script intelligence metadata
    prep_result["metadata"]["script_detection"] = script_info
    prep_result["metadata"]["script_detection_time_ms"] = script_time
    prep_result["metadata"]["devanagari_recovery"] = devanagari_recovery
    prep_result["metadata"]["translation_blocked"] = translation_blocked
    prep_result["metadata"]["review_gate"] = review_gate

    if annotated_img_path:
        prep_result["metadata"]["annotated_image"] = f"/debug/{annotated_img_path}"

    if not raw_text.strip():
        failure_reasons.add("Empty OCR Result")
        raise RuntimeError(
            "OCR produced no readable text. Paddle returned no parseable text blocks and "
            "Tesseract fallback did not produce text. Check PaddleOCR result parsing, "
            "install/configure Tesseract, or route this sample to human verification."
        )

    total_time = round((time.time() - overall_start) * 1000, 2)
    logger.info("[END] OCR pipeline execution (total time=%.2fms)", total_time)
    logger.info(
        "Devanagari recovery | score=%.4f ratio=%.4f token_ratio=%.4f translation_blocked=%s",
        devanagari_recovery["devanagari_recovery_score"],
        devanagari_recovery["devanagari_ratio"],
        devanagari_recovery["devanagari_token_ratio"],
        translation_blocked,
    )

    return {
        "text": raw_text,
        "raw_text": raw_text,
        "ocr_confidence": round(avg_conf, 4),
        "confidence": round(avg_conf, 4),
        "ocr_engine": "ensemble",
        "uncertain_regions": uncertain_regions,
        "officer_review_regions": officer_review_regions,
        "line_results": line_results,
        "preprocessing_metadata": prep_result["metadata"],
        "processing_time_ms": total_time,
        # --- New fields for Devanagari intelligence ---
        "script_detection": script_info,
        "devanagari_recovery": devanagari_recovery,
        "translation_blocked": translation_blocked,
        "review_gate": review_gate,
    }
