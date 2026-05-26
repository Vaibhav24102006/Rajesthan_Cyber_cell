import cv2
import numpy as np
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)

def deskew_image(thresh: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
    """
    Detects tilt and deskews the image.
    Uses the thresholded image to find text blocks and calculates the skew angle.
    """
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img_bgr

    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # If angle is tiny or implausibly large, skip rotation. The min-area rectangle
    # often reports near-90-degree angles on handwritten pages and can rotate a
    # valid landscape complaint into unusable vertical text strips.
    if abs(angle) < 0.5 or abs(angle) > 10.0:
        logger.info("Skipping deskew rotation; detected angle %.2f is outside safe range.", angle)
        return img_bgr

    (h, w) = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def upscale_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Upscales the image using cubic interpolation.
    """
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _unsharp_mask(gray: np.ndarray, sigma: float = 1.0, amount: float = 1.4) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def _blue_ink_enhance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Builds a high-contrast grayscale image that favors blue/purple handwritten ink.
    The output stays black-on-white for OCR recognizers.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 35, 20])
    upper_blue = np.array([145, 255, 220])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    paper = np.full(mask.shape, 255, dtype=np.uint8)
    paper[mask > 0] = 20
    return paper

def analyze_image_heuristics(img_gray: np.ndarray) -> dict:
    """
    Calculates heuristic metrics for dynamic preprocessing and failure classification.
    """
    # 1. Contrast (Standard deviation of pixel intensities)
    contrast_score = np.std(img_gray)
    
    # 2. Noise Density (Laplacian variance)
    noise_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    
    # 3. Stroke Thickness Estimate (Morphological gradient mean)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
    stroke_thickness = np.mean(gradient)
    
    # 4. Skew Severity Estimate
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Only look at mostly horizontal lines
                angles.append(angle)
    skew_angle = np.median(angles) if angles else 0.0

    # Determine recommended profile
    recommended_profile = "adaptive"
    if noise_var < 50:  # Very blurred or noisy
        recommended_profile = "denoise-heavy"
    elif contrast_score < 35: # Low contrast
        recommended_profile = "contrast-heavy"
    elif contrast_score > 70 and noise_var > 1000: # High contrast but maybe sharp noise
        recommended_profile = "otsu"
        
    return {
        "contrast_score": round(float(contrast_score), 2),
        "noise_density": round(float(noise_var), 2),
        "stroke_thickness": round(float(stroke_thickness), 2),
        "skew_angle": round(float(skew_angle), 2),
        "recommended_profile": recommended_profile
    }

def preprocess_for_ocr(img_bgr: np.ndarray, profile: str = "auto", debug_dir: Path = None, file_id: str = "doc") -> dict:
    """
    Applies advanced preprocessing based on profile:
    - auto: Dynamically select based on heuristics
    - adaptive (default): Good for most handwritten documents.
    - otsu: Better for consistent contrast documents.
    - denoise-heavy: For noisy scans or camera photos.
    - contrast-heavy: For faint handwriting.
    - sharpen-heavy: To bring out edges in blurry text.
    - handwriting-clahe: CLAHE + edge-preserving smoothing + unsharp mask.
    - handwriting-blue-ink: Blue/purple ink isolation for ballpoint handwritten complaints.
    - handwriting-fast: Lightweight CLAHE/unsharp profile for field latency.
    - handwriting-devanagari: Devanagari handwriting recovery; matra preservation, stroke enhancement,
      bilateral filtering, morphological close/open for glyph continuity, directional dilation,
      and adaptive threshold tuned for Devanagari handwritten glyph structure.
    """
    start_time = time.time()
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    heuristics = analyze_image_heuristics(gray)
    
    if profile == "auto":
        profile = heuristics["recommended_profile"]
        
    metadata = {"profile": profile, "heuristics": heuristics}
    
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{file_id}_0_raw.jpg"), img_bgr)

    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    ocr_input_bgr = img_bgr

    # 2. Profile-specific enhancements
    if profile == "denoise-heavy":
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Non-local means denoising
        enhanced = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)
    elif profile == "contrast-heavy":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
    elif profile == "sharpen-heavy":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(blur, -1, kernel)
    elif profile == "handwriting-clahe":
        smooth = cv2.bilateralFilter(gray, 7, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        enhanced = _unsharp_mask(clahe.apply(smooth), sigma=1.0, amount=1.2)
        ocr_input_bgr = _gray_to_bgr(enhanced)
    elif profile == "handwriting-blue-ink":
        enhanced = _blue_ink_enhance(img_bgr)
        enhanced = _unsharp_mask(enhanced, sigma=0.8, amount=0.8)
        ocr_input_bgr = _gray_to_bgr(enhanced)
    elif profile == "handwriting-fast":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = _unsharp_mask(clahe.apply(gray), sigma=0.8, amount=0.9)
        ocr_input_bgr = _gray_to_bgr(enhanced)
    elif profile == "handwriting-devanagari":
        smooth = cv2.bilateralFilter(gray, 9, 60, 60)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(smooth)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        enhanced = cv2.dilate(enhanced, kernel_dilate, iterations=1)
        enhanced = _unsharp_mask(enhanced, sigma=0.6, amount=1.0)
        ocr_input_bgr = _gray_to_bgr(enhanced)
    else:
        enhanced = cv2.GaussianBlur(gray, (5, 5), 0)
        
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{file_id}_1_{profile}_enhanced.jpg"), enhanced)

    # 3. Thresholding
    if profile == "otsu":
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif profile == "handwriting-devanagari":
        blur_for_thresh = cv2.GaussianBlur(enhanced, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur_for_thresh,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            6
        )
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    else:
        # Default is adaptive
        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            11
        )
        
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{file_id}_2_{profile}_thresh.jpg"), thresh)

    # 4. Deskew
    deskewed_bgr = deskew_image(thresh, ocr_input_bgr)
    
    # Re-apply threshold on deskewed image for segmentation
    deskewed_gray = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
    if profile == "otsu":
        _, deskewed_thresh = cv2.threshold(deskewed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif profile == "handwriting-devanagari":
        deskewed_blur = cv2.GaussianBlur(deskewed_gray, (3, 3), 0)
        deskewed_thresh = cv2.adaptiveThreshold(deskewed_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 6)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        deskewed_thresh = cv2.morphologyEx(deskewed_thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    else:
        deskewed_blur = cv2.GaussianBlur(deskewed_gray, (5, 5), 0)
        deskewed_thresh = cv2.adaptiveThreshold(deskewed_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 11)
    
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{file_id}_3_{profile}_deskewed.jpg"), deskewed_bgr)

    # 5. Optional upscale. Full-page OCR is CPU-heavy, so low-memory mode keeps
    # the native resolution unless explicitly tuned upward.
    try:
        upscale_factor = float(os.getenv("OCR_PREPROCESS_UPSCALE_FACTOR", "1.0"))
    except Exception:
        upscale_factor = 1.0
    upscale_factor = max(1.0, min(upscale_factor, 2.0))
    if upscale_factor > 1.0:
        upscaled_bgr = upscale_image(deskewed_bgr, scale=upscale_factor)
        upscaled_thresh = upscale_image(deskewed_thresh, scale=upscale_factor)
    else:
        upscaled_bgr = deskewed_bgr
        upscaled_thresh = deskewed_thresh
    metadata["upscale_factor"] = upscale_factor
    
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{file_id}_4_{profile}_upscaled.jpg"), upscaled_bgr)

    metadata["preprocessing_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return {
        "processed_bgr": upscaled_bgr,
        "processed_thresh": upscaled_thresh,
        "metadata": metadata
    }

def annotate_image(img_bgr: np.ndarray, line_results: list, debug_dir: Path = None, file_id: str = "doc") -> str:
    """
    Draws bounding boxes based on line_results confidence.
    Green > 0.8, Yellow > 0.5, Red <= 0.5.
    """
    annotated = img_bgr.copy()
    
    for res in line_results:
        y1, y2 = res.get("y_coords", (0, 0))
        # Account for upscale if coords are from upscaled image, but since we draw on upscaled_bgr it's fine
        conf = res.get("confidence", 0.0)
        if conf > 0.8:
            color = (0, 255, 0) # Green
        elif conf > 0.5:
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 0, 255) # Red
            
        cv2.rectangle(annotated, (0, y1), (annotated.shape[1], y2), color, 2)
        # Put confidence text
        cv2.putText(annotated, f"{conf:.2f} {res.get('engine','')}", (10, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if debug_dir:
        out_path = debug_dir / f"{file_id}_final_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
        return out_path.name
    return ""
