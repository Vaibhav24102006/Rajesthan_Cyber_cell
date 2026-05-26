import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np

from app.services.ocr.pipeline import extract_text_from_document_pipeline
from app.services.reconstruction.language_detection import detect_regional_language

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
HINDI_STOPWORDS = {"mera", "mere", "mujhe", "maine", "paise", "rupaye", "khata", "bank", "se",
                    "kat", "gaye", "nikal", "gaya", "dhokha", "shikayat", "kripya", "karyavahi",
                    "police", "thana", "nivedan", "mahoday", "shriman", "vishey", "praye"}


DEFAULT_PROFILES = [
    "auto",
    "adaptive",
    "contrast-heavy",
    "sharpen-heavy",
    "handwriting-fast",
    "handwriting-clahe",
    "handwriting-blue-ink",
    "handwriting-devanagari",
]


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _token_recovery(text: str, expected_tokens: list[str]) -> tuple[float, list[str]]:
    normalized = _norm_text(text)
    missing = [token for token in expected_tokens if _norm_text(token) not in normalized]
    recovered = len(expected_tokens) - len(missing)
    rate = recovered / len(expected_tokens) if expected_tokens else 1.0
    return round(rate, 4), missing


def _gibberish_density(text: str) -> float:
    tokens = re.findall(r"[\w@\-/₹,.]+", text or "", flags=re.UNICODE)
    if not tokens:
        return 1.0
    bad = 0
    for token in tokens:
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", token))
        has_digit = bool(re.search(r"\d", token))
        has_latin_word = bool(re.search(r"[A-Za-z]{2,}", token))
        if not (has_devanagari or has_digit or has_latin_word):
            bad += 1
        elif len(token) >= 8 and not has_digit and not has_devanagari and SequenceMatcher(None, token.lower(), "transaction").ratio() < 0.25:
            bad += 1
    return round(bad / len(tokens), 4)


def _heatmap(img_bgr, line_results: list[dict], out_path: Path) -> None:
    overlay = img_bgr.copy()
    for item in line_results:
        conf = float(item.get("confidence") or 0.0)
        if conf >= 0.75:
            continue
        box = item.get("box") or []
        if len(box) >= 4:
            poly = np.array(box, dtype=np.int32)
            color = (0, 0, 255) if conf < 0.50 else (0, 165, 255)
            cv2.polylines(overlay, [poly], True, color, 3)
            x, y = poly[0]
            cv2.putText(overlay, f"{conf:.2f}", (int(x), max(20, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            y1, y2 = item.get("y_coords", (0, 0))
            color = (0, 0, 255) if conf < 0.50 else (0, 165, 255)
            cv2.rectangle(overlay, (0, int(y1)), (overlay.shape[1], int(y2)), color, 3)
    cv2.imwrite(str(out_path), overlay)


def _devanagari_recovery_score(text: str) -> dict:
    if not text.strip():
        return {"devanagari_ratio": 0.0, "devanagari_token_ratio": 0.0,
                "hindi_stopword_hits": 0, "devanagari_recovery_metric": 0.0}
    total_chars = len(text.replace(" ", ""))
    dev_chars = len(DEVANAGARI_RE.findall(text))
    dev_ratio = dev_chars / max(1, total_chars)
    tokens = re.findall(r"[\w\u0900-\u097F]+", text)
    dev_tokens = sum(1 for t in tokens if DEVANAGARI_RE.search(t)) if tokens else 0
    dev_token_ratio = dev_tokens / max(1, len(tokens))
    text_lower = text.lower()
    stopword_hits = sum(1 for sw in HINDI_STOPWORDS if sw in text_lower)
    combined = dev_ratio * 0.4 + dev_token_ratio * 0.3 + min(1.0, stopword_hits / 10.0) * 0.3
    return {
        "devanagari_ratio": round(dev_ratio, 4),
        "devanagari_token_ratio": round(dev_token_ratio, 4),
        "hindi_stopword_hits": stopword_hits,
        "devanagari_recovery_metric": round(combined, 4),
    }


def _score_result(raw_text: str, ground_truth: str, expected_tokens: list[str], confidence: float, runtime_ms: float) -> dict:
    similarity = SequenceMatcher(None, _norm_text(raw_text), _norm_text(ground_truth)).ratio()
    token_rate, missing_tokens = _token_recovery(raw_text, expected_tokens)
    gibberish_density = _gibberish_density(raw_text)
    runtime_score = max(0.0, 1.0 - max(0.0, runtime_ms - 15000.0) / 60000.0)
    dev_recovery = _devanagari_recovery_score(raw_text)
    score = (
        similarity * 0.20
        + token_rate * 0.20
        + dev_recovery["devanagari_recovery_metric"] * 0.20
        + float(confidence or 0.0) * 0.15
        + (1.0 - gibberish_density) * 0.10
        + runtime_score * 0.05
    )
    return {
        "score": round(score, 4),
        "ground_truth_similarity": round(similarity, 4),
        "token_recovery_rate": token_rate,
        "missing_tokens": missing_tokens,
        "gibberish_density": gibberish_density,
        "runtime_score": round(runtime_score, 4),
        "devanagari_recovery": dev_recovery,
    }


def main() -> int:
    case_path = ROOT / "tests" / "ocr_benchmark_ground_truth.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    image_path = Path(case["image_path"])
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise SystemExit(f"Unable to read benchmark image: {image_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "storage" / "datasets" / "ocr_adaptation" / case["id"] / run_id
    debug_dir = run_dir / "debug"
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    archived_image = run_dir / image_path.name
    if not archived_image.exists():
        shutil.copy2(image_path, archived_image)

    profile_arg = os.getenv("OCR_BENCHMARK_PROFILES", "")
    profiles = [p.strip() for p in profile_arg.split(",") if p.strip()] or DEFAULT_PROFILES
    lang_arg = os.getenv("OCR_BENCHMARK_LANGS", "")
    langs = [p.strip() for p in lang_arg.split(",") if p.strip()] or ["en", "hi"]
    max_side_arg = os.getenv("OCR_BENCHMARK_MAX_SIDES", "")
    max_sides = [int(p.strip()) for p in max_side_arg.split(",") if p.strip()] or [1280]

    results = []
    for lang in langs:
        for max_side in max_sides:
            for profile in profiles:
                combo_id = f"{profile}_{lang}_{max_side}".replace("-", "_")
                combo_debug = debug_dir / combo_id
                start = perf_counter()
                try:
                    result = extract_text_from_document_pipeline(
                        img_bgr,
                        file_id=combo_id,
                        debug_dir=combo_debug,
                        fallback_threshold=0.5,
                        preprocessing_profile=profile,
                        paddle_lang=lang,
                        paddle_max_side=max_side,
                    )
                    runtime_ms = round((perf_counter() - start) * 1000, 2)
                    raw_text = result.get("raw_text", "")
                    detected_language, detected_dialect, lang_conf = detect_regional_language(
                        raw_text,
                        result.get("confidence", 0.0),
                    )
                    metrics = _score_result(
                        raw_text,
                        case["expected_readable_ground_truth"],
                        case["must_recover_tokens"],
                        result.get("confidence", 0.0),
                        runtime_ms,
                    )
                    heatmap_path = combo_debug / f"{combo_id}_failed_region_heatmap.jpg"
                    _heatmap(img_bgr, result.get("line_results", []), heatmap_path)

                    script_det = result.get("script_detection", {})
                    dev_recov = result.get("devanagari_recovery", {})
                    review_gate = result.get("review_gate", {})
                    trans_blocked = result.get("translation_blocked", False)

                    results.append(
                        {
                            "profile": profile,
                            "paddle_lang": lang,
                            "paddle_max_side": max_side,
                            "raw_ocr_output": raw_text,
                            "ocr_confidence": result.get("confidence", 0.0),
                            "detected_language": detected_language,
                            "detected_dialect": detected_dialect,
                            "language_confidence": lang_conf,
                            "script_detection": {
                                "script": script_det.get("script"),
                                "confidence": script_det.get("confidence"),
                                "scores": script_det.get("scores"),
                            },
                            "devanagari_recovery": dev_recov,
                            "translation_blocked": trans_blocked,
                            "review_gate": review_gate,
                            "failed_token_regions": [
                                item for item in result.get("line_results", []) if float(item.get("confidence") or 0.0) < 0.75
                            ],
                            "officer_review_regions": result.get("officer_review_regions", []),
                            "bounding_boxes": result.get("line_results", []),
                            "preprocessing_profile_used": result.get("preprocessing_metadata", {}).get("profile", profile),
                            "preprocessing_metadata": result.get("preprocessing_metadata", {}),
                            "runtime_ms": runtime_ms,
                            "heatmap_path": str(heatmap_path),
                            "metrics": metrics,
                            "failure_reasons": result.get("preprocessing_metadata", {}).get("failure_reasons", []),
                        }
                    )
                except Exception as exc:
                    runtime_ms = round((perf_counter() - start) * 1000, 2)
                    results.append(
                        {
                            "profile": profile,
                            "paddle_lang": lang,
                            "paddle_max_side": max_side,
                            "raw_ocr_output": "",
                            "ocr_confidence": 0.0,
                            "runtime_ms": runtime_ms,
                            "metrics": {"score": 0.0},
                            "failure_reasons": [str(exc)],
                        }
                    )

    best = max(results, key=lambda item: item.get("metrics", {}).get("score", 0.0))
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "uploaded_image": str(archived_image),
        "performance_targets": {
            "ocr_runtime_ms_max": 15000,
            "primary_goal": "recover readable native-language structure before translation",
        },
        "best_profile": {
            "profile": best.get("profile"),
            "paddle_lang": best.get("paddle_lang"),
            "paddle_max_side": best.get("paddle_max_side"),
            "score": best.get("metrics", {}).get("score", 0.0),
            "ocr_confidence": best.get("ocr_confidence", 0.0),
            "runtime_ms": best.get("runtime_ms", 0.0),
            "token_recovery_rate": best.get("metrics", {}).get("token_recovery_rate", 0.0),
            "devanagari_recovery_metric": best.get("metrics", {}).get("devanagari_recovery", {}).get("devanagari_recovery_metric", 0.0),
            "devanagari_ratio": best.get("metrics", {}).get("devanagari_recovery", {}).get("devanagari_ratio", 0.0),
            "script_detection": best.get("script_detection", {}),
            "translation_blocked": best.get("translation_blocked", False),
            "missing_tokens": best.get("metrics", {}).get("missing_tokens", []),
            "heatmap_path": best.get("heatmap_path"),
        },
        "results": results,
    }
    (run_dir / "ocr_adaptation_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_config_path = ROOT / "storage" / "datasets" / "ocr_adaptation" / case["id"] / "best_profile.json"
    should_update_best = True
    if best_config_path.exists():
        try:
            existing_best = json.loads(best_config_path.read_text(encoding="utf-8"))
            should_update_best = summary["best_profile"]["score"] >= float(existing_best.get("score", 0.0))
        except Exception:
            should_update_best = True
    if should_update_best:
        best_config_path.write_text(json.dumps(summary["best_profile"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_path": str(run_dir / "ocr_adaptation_results.json"), "best_profile": summary["best_profile"]}, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
