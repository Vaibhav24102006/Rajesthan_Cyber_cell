import numpy as np
import cv2
import logging
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def segment_lines(thresh_img: np.ndarray, bgr_img: np.ndarray, debug_dir: Path = None, file_id: str = "doc") -> List[np.ndarray]:
    """
    Segments the image into horizontal text lines using horizontal projection.
    Returns a list of BGR line images.
    """
    # Sum along rows
    horizontal_projection = np.sum(thresh_img, axis=1)
    
    # Threshold the projection to find empty rows (gaps) vs text rows
    # Max value per pixel in thresh is 255.
    # If the sum of a row is very low, it's considered empty.
    max_val = np.max(horizontal_projection)
    if max_val == 0:
        return [{"image": bgr_img, "coords": (0, bgr_img.shape[0])}] # No text found, just return original

    # We need to find continuous segments of text
    threshold = max_val * 0.05 # 5% of max
    is_text_row = horizontal_projection > threshold
    
    lines = []
    in_line = False
    start_y = 0
    
    for y, is_text in enumerate(is_text_row):
        if is_text and not in_line:
            in_line = True
            start_y = y
        elif not is_text and in_line:
            in_line = False
            end_y = y
            # Ignore lines that are too small (noise)
            if end_y - start_y > 10: 
                # Add some padding
                pad = 10
                y1 = max(0, start_y - pad)
                y2 = min(bgr_img.shape[0], end_y + pad)
                lines.append((y1, y2))
                
    # If it ends while still in a line
    if in_line:
        end_y = len(is_text_row)
        if end_y - start_y > 10:
            pad = 10
            y1 = max(0, start_y - pad)
            y2 = min(bgr_img.shape[0], end_y + pad)
            lines.append((y1, y2))

    if not lines:
        return [{"image": bgr_img, "coords": (0, bgr_img.shape[0])}]

    line_data = []
    
    if debug_dir:
        debug_img = bgr_img.copy()
        
    for i, (y1, y2) in enumerate(lines):
        # Extract the line
        line_img = bgr_img[y1:y2, :]
        line_data.append({"image": line_img, "coords": (y1, y2)})
        
        if debug_dir:
            cv2.rectangle(debug_img, (0, y1), (bgr_img.shape[1], y2), (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / f"{file_id}_5_line_{i}.jpg"), line_img)
            
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{file_id}_5_segments.jpg"), debug_img)

    return line_data
