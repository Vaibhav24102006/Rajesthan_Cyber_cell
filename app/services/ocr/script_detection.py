import logging
import numpy as np
import cv2
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def _compute_connected_component_stats(binary: np.ndarray) -> Dict:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = []
    aspect_ratios = []
    extents = []
    widths = []
    heights = []
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 15 or a > 50000:
            continue
        areas.append(a)
        widths.append(w)
        heights.append(h)
        if h > 0:
            aspect_ratios.append(w / float(h))
        extent = a / float(w * h) if w * h > 0 else 0
        extents.append(extent)
    if not areas:
        return {"cc_count": 0, "mean_area": 0, "mean_ar": 0, "mean_extent": 0, "std_ar": 0,
                "mean_width": 0, "mean_height": 0, "std_width": 0, "std_height": 0,
                "wide_ratio": 0, "tall_ratio": 0}
    return {
        "cc_count": len(areas),
        "mean_area": float(np.mean(areas)),
        "mean_ar": float(np.mean(aspect_ratios)) if aspect_ratios else 0,
        "mean_extent": float(np.mean(extents)) if extents else 0,
        "std_ar": float(np.std(aspect_ratios)) if aspect_ratios else 0,
        "mean_width": float(np.mean(widths)) if widths else 0,
        "mean_height": float(np.mean(heights)) if heights else 0,
        "std_width": float(np.std(widths)) if widths else 0,
        "std_height": float(np.std(heights)) if heights else 0,
        "wide_ratio": sum(1 for ar in aspect_ratios if ar > 1.5) / len(aspect_ratios) if aspect_ratios else 0,
        "tall_ratio": sum(1 for ar in aspect_ratios if ar < 0.5) / len(aspect_ratios) if aspect_ratios else 0,
    }


def _horizontal_projection_features(binary: np.ndarray) -> Dict:
    h_proj = np.sum(binary > 0, axis=1)
    if np.max(h_proj) == 0:
        return {"peak_sharpness": 0, "peak_position": 0.5, "valley_depth": 0, "profile_variance": 0}
    h_proj_norm = h_proj / float(np.max(h_proj))
    mid = len(h_proj_norm) // 2
    peak_val = float(np.max(h_proj_norm))
    peak_pos = float(np.argmax(h_proj_norm)) / max(1, len(h_proj_norm) - 1)
    top_third = h_proj_norm[:len(h_proj_norm)//3]
    bottom_third = h_proj_norm[2*len(h_proj_norm)//3:]
    top_mean = float(np.mean(top_third)) if len(top_third) > 0 else 0
    bot_mean = float(np.mean(bottom_third)) if len(bottom_third) > 0 else 0
    valley_depth = peak_val - min(top_mean, bot_mean)
    sharpness = float(np.std(h_proj_norm))
    return {
        "peak_sharpness": sharpness,
        "peak_position": peak_pos,
        "valley_depth": valley_depth,
        "profile_variance": float(np.var(h_proj_norm)),
        "top_density": top_mean,
        "bottom_density": bot_mean,
        "mid_density": float(np.mean(h_proj_norm[mid-10:mid+10])) if mid >= 10 and mid+10 <= len(h_proj_norm) else 0,
    }


def _shirorekha_detection(binary: np.ndarray, cc_stats: Dict) -> float:
    h_proj = np.sum(binary > 0, axis=1)
    if np.max(h_proj) == 0 or cc_stats["cc_count"] < 5:
        return 0.0
    h_proj_norm = h_proj / float(np.max(h_proj))
    strong_rows = np.where(h_proj_norm > 0.45)[0]
    if len(strong_rows) < 2:
        return 0.0
    gaps = np.diff(strong_rows)
    continuous_segments = np.split(strong_rows, np.where(gaps > 2)[0] + 1)
    long_segments = [seg for seg in continuous_segments if len(seg) > 3]
    if not long_segments:
        return 0.0
    max_seg_len = max(len(seg) for seg in long_segments)
    row_coverage = max_seg_len / max(1, len(h_proj_norm))
    thinness_ratio = 0.0
    for seg in long_segments:
        thin = 1.0 - min(1.0, (seg[-1] - seg[0]) / (cc_stats["mean_height"] * 3 + 1))
        if thin > thinness_ratio:
            thinness_ratio = thin
    shirorekha_score = (row_coverage * 0.5 + thinness_ratio * 0.5)
    return float(min(1.0, shirorekha_score * 2.0))


def _vertical_profile_features(binary: np.ndarray) -> Dict:
    v_proj = np.sum(binary > 0, axis=0)
    if np.max(v_proj) == 0:
        return {"v_profile_variance": 0, "v_peak_sharpness": 0, "column_std": 0}
    v_proj_norm = v_proj / float(np.max(v_proj))
    return {
        "v_profile_variance": float(np.var(v_proj_norm)),
        "v_peak_sharpness": float(np.std(v_proj_norm)),
        "column_std": float(np.std(np.where(v_proj_norm > 0.3)[0])) if np.any(v_proj_norm > 0.3) else 0,
    }


def _contour_morphology(binary: np.ndarray) -> Dict:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"num_contours": 0, "mean_contour_area": 0, "mean_perimeter": 0,
                "mean_circularity": 0, "complexity": 0}
    areas = []
    perimeters = []
    circularities = []
    for c in contours:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if a < 10:
            continue
        areas.append(float(a))
        perimeters.append(float(p))
        circ = 4 * np.pi * a / (p * p + 1e-6)
        circularities.append(min(circ, 1.0))
    if not areas:
        return {"num_contours": 0, "mean_contour_area": 0, "mean_perimeter": 0,
                "mean_circularity": 0, "complexity": 0}
    return {
        "num_contours": len(areas),
        "mean_contour_area": float(np.mean(areas)),
        "mean_perimeter": float(np.mean(perimeters)) if perimeters else 0,
        "mean_circularity": float(np.mean(circularities)) if circularities else 0,
        "complexity": float(np.std(perimeters) / (np.mean(perimeters) + 1e-6)) if perimeters else 0,
    }


def _detect_grid_structure(binary: np.ndarray) -> float:
    h, w = binary.shape
    cell_h, cell_w = max(1, h // 20), max(1, w // 20)
    filled_cells = 0
    total_cells = 0
    for y in range(0, h, cell_h):
        for x in range(0, w, cell_w):
            tile = binary[y:min(y+cell_h, h), x:min(x+cell_w, w)]
            if tile.size > 0 and np.sum(tile > 0) / tile.size > 0.05:
                filled_cells += 1
            total_cells += 1
    return filled_cells / max(1, total_cells)


def _estimate_numeric_density(binary: np.ndarray) -> float:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    small_regular = 0
    total_significant = 0
    for c in contours:
        a = cv2.contourArea(c)
        if a < 20 or a > 2000:
            continue
        total_significant += 1
        x, y, w, h = cv2.boundingRect(c)
        if 8 <= w <= 40 and 12 <= h <= 50:
            extent = a / (w * h + 1e-6)
            if 0.3 <= extent <= 0.9:
                small_regular += 1
    if total_significant == 0:
        return 0.0
    return small_regular / total_significant


def classify_script_visually(img_bgr: np.ndarray) -> Dict:
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cc_stats = _compute_connected_component_stats(binary)
    h_proj = _horizontal_projection_features(binary)
    v_proj = _vertical_profile_features(binary)
    shirorekha_score = _shirorekha_detection(binary, cc_stats)
    morph = _contour_morphology(binary)
    grid_density = _detect_grid_structure(binary)
    numeric_density = _estimate_numeric_density(binary)

    features = {
        "cc_aspect_ratio_mean": cc_stats["mean_ar"],
        "cc_aspect_ratio_std": cc_stats["std_ar"],
        "cc_height_std": cc_stats["std_height"],
        "cc_width_std": cc_stats["std_width"],
        "cc_tall_ratio": cc_stats["tall_ratio"],
        "cc_wide_ratio": cc_stats["wide_ratio"],
        "cc_count": cc_stats["cc_count"],
        "h_proj_peak_sharpness": h_proj["peak_sharpness"],
        "h_proj_peak_position": h_proj["peak_position"],
        "h_proj_valley_depth": h_proj["valley_depth"],
        "h_proj_top_density": h_proj["top_density"],
        "h_proj_mid_density": h_proj["mid_density"],
        "shirorekha_score": shirorekha_score,
        "v_profile_variance": v_proj["v_profile_variance"],
        "contour_complexity": morph["complexity"],
        "mean_circularity": morph["mean_circularity"],
        "grid_density": grid_density,
        "numeric_density": numeric_density,
    }

    devanagari_score = (
        shirorekha_score * 0.35
        + min(1.0, h_proj["valley_depth"] * 2.0) * 0.15
        + min(1.0, h_proj["mid_density"] * 1.5) * 0.10
        + (1.0 - min(1.0, morph["mean_circularity"] * 2.0)) * 0.10
        + min(1.0, cc_stats["std_height"] / 20.0) * 0.10
        - min(1.0, cc_stats["tall_ratio"]) * 0.05
        + min(1.0, features["cc_aspect_ratio_mean"] * 0.5) * 0.10
        + features["grid_density"] * 0.05
    )
    latin_score = (
        min(1.0, cc_stats["tall_ratio"] * 2.0) * 0.20
        + morph["mean_circularity"] * 0.20
        + (1.0 - shirorekha_score) * 0.20
        + (1.0 - h_proj["mid_density"]) * 0.10
        + min(1.0, morph["complexity"] * 0.5) * 0.15
        + min(1.0, cc_stats["std_width"] / 30.0) * 0.15
    )

    devanagari_score = max(0.0, min(1.0, devanagari_score))
    latin_score = max(0.0, min(1.0, latin_score))
    numeric_score = max(0.0, min(1.0, numeric_density * 2.0))

    if devanagari_score > latin_score * 1.3 and devanagari_score > 0.25:
        script = "devanagari"
        confidence = devanagari_score
    elif latin_score > devanagari_score * 1.3 and latin_score > 0.25:
        script = "latin"
        confidence = latin_score
    elif abs(devanagari_score - latin_score) < 0.15 and max(devanagari_score, latin_score) > 0.3:
        script = "mixed"
        confidence = max(devanagari_score, latin_score)
    elif numeric_score > 0.50 and max(devanagari_score, latin_score) < 0.30:
        script = "numeric_heavy"
        confidence = numeric_score
    else:
        script = "latin"
        confidence = max(0.3, latin_score)

    if numeric_score > 0.60 and script != "numeric_heavy":
        if confidence < 0.50:
            script = "mixed"
            confidence = max(confidence, numeric_score)

    logger.info(
        "Visual script classification: script=%s confidence=%.4f "
        "devanagari_score=%.4f latin_score=%.4f numeric_score=%.4f shirorekha=%.4f",
        script, confidence, devanagari_score, latin_score, numeric_score, shirorekha_score
    )

    return {
        "script": script,
        "confidence": round(confidence, 4),
        "scores": {
            "devanagari": round(devanagari_score, 4),
            "latin": round(latin_score, 4),
            "numeric": round(numeric_score, 4),
        },
        "features": features,
    }
