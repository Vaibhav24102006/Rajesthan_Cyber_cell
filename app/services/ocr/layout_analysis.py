import numpy as np
import cv2
import logging
from typing import List, Dict, Any
from pathlib import Path

from app.services.ocr.segmentation import segment_lines

logger = logging.getLogger(__name__)

def classify_region_heuristics(thresh_region: np.ndarray, y1: int, y2: int, total_height: int) -> Dict[str, Any]:
    """
    Analyzes a region and heuristically classifies it.
    Returns dict with type and confidence.
    """
    height, width = thresh_region.shape
    if height == 0 or width == 0:
        return {"type": "paragraph", "confidence": 0.5}

    pixel_density = np.count_nonzero(thresh_region) / (height * width)
    aspect_ratio = width / float(height)
    
    # Calculate relative position on the page (0.0 is top, 1.0 is bottom)
    center_y = (y1 + y2) / 2.0
    rel_y = center_y / total_height if total_height > 0 else 0.5
    
    # Find contours within the region to estimate complexity
    contours, _ = cv2.findContours(thresh_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    
    if num_contours == 0:
         return {"type": "paragraph", "confidence": 0.5}
         
    # Signatures are often sparse, high aspect ratio, few large contours, near bottom
    if rel_y > 0.7 and pixel_density < 0.15 and aspect_ratio > 2.0 and num_contours < 15:
        # Check if the contours are large
        areas = [cv2.contourArea(c) for c in contours]
        if areas and max(areas) > 100:
            return {"type": "signature", "confidence": 0.8}
            
    # Stamps are often dense, roughly square or circular (aspect ratio near 1.0)
    if pixel_density > 0.3 and 0.8 < aspect_ratio < 1.5:
        return {"type": "stamp", "confidence": 0.7}
        
    # Headers are usually at the top, small height relative to page
    if rel_y < 0.15 and height < (total_height * 0.1):
        return {"type": "header", "confidence": 0.9}

    # Default to paragraph (standard text line)
    return {"type": "paragraph", "confidence": 0.9}


def analyze_layout(thresh_img: np.ndarray, bgr_img: np.ndarray, debug_dir: Path = None, file_id: str = "doc") -> List[Dict[str, Any]]:
    """
    Segments the image into regions and classifies them.
    Currently delegates segmentation to horizontal projection, then applies heuristic classification.
    """
    total_height = thresh_img.shape[0]
    
    # 1. Reuse line segmentation to get bounding boxes (we treat lines as regions for now)
    regions_data = segment_lines(thresh_img, bgr_img, debug_dir, file_id)
    
    analyzed_regions = []
    
    if debug_dir:
        debug_img = bgr_img.copy()
        
    for i, data in enumerate(regions_data):
        line_img = data["image"]
        y1, y2 = data["coords"]
        
        # Extract the same crop from the thresholded image for analysis
        thresh_crop = thresh_img[y1:y2, :]
        
        # Classify the region
        classification = classify_region_heuristics(thresh_crop, y1, y2, total_height)
        
        region_info = {
            "image": line_img,
            "coords": (y1, y2),
            "region_type": classification["type"],
            "region_confidence": classification["confidence"]
        }
        analyzed_regions.append(region_info)
        
        if debug_dir:
            color = (0, 255, 0)
            if classification["type"] == "signature":
                color = (255, 0, 0) # Blue
            elif classification["type"] == "stamp":
                color = (0, 0, 255) # Red
            elif classification["type"] == "header":
                color = (255, 255, 0) # Cyan
                
            cv2.rectangle(debug_img, (0, y1), (bgr_img.shape[1], y2), color, 2)
            cv2.putText(debug_img, f"{classification['type']} ({classification['confidence']:.2f})", 
                       (10, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{file_id}_5_layout.jpg"), debug_img)
        
    return analyzed_regions
