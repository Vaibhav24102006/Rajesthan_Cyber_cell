import os
import sys
import time
import json
import csv
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.ocr.pipeline import extract_text_from_document_pipeline
import cv2

def run_benchmark(test_dir: str, out_dir: str):
    test_path = Path(test_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not test_path.exists() or not test_path.is_dir():
        print(f"Test directory {test_dir} not found.")
        return
        
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in test_path.iterdir() if f.suffix.lower() in valid_exts]
    
    if not images:
        print("No images found in test directory.")
        return
        
    print(f"Found {len(images)} images to benchmark.")
    
    csv_file = out_path / "benchmark_results.csv"
    json_file = out_path / "benchmark_results.json"
    
    results = []
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Confidence", "Time_ms", "Total_Lines", "Uncertain_Density", 
            "Paddle_Count", "Tesseract_Count", "Retries_Total"
        ])
        
        for img_path in images:
            print(f"Processing {img_path.name}...")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"Failed to read {img_path.name}")
                continue
                
            debug_dir = out_path / img_path.stem
            
            try:
                result = extract_text_from_document_pipeline(
                    img_bgr, 
                    file_id=img_path.stem, 
                    debug_dir=debug_dir, 
                    fallback_threshold=0.6
                )
                
                total_lines = len(result["line_results"])
                retries_total = sum(r.get("retries", 0) for r in result["line_results"])
                uncertain_density = len(result["uncertain_regions"]) / total_lines if total_lines else 0
                
                engine_usage = result["preprocessing_metadata"].get("engine_usage", {})
                
                row = [
                    img_path.name,
                    result["confidence"],
                    result["processing_time_ms"],
                    total_lines,
                    uncertain_density,
                    engine_usage.get("paddle", 0),
                    engine_usage.get("tesseract", 0),
                    retries_total
                ]
                writer.writerow(row)
                f.flush()
                
                result["filename"] = img_path.name
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Benchmarking complete. Results saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Benchmarking Suite")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--out-dir", type=str, default="benchmark_output", help="Output directory for metrics and debug images")
    
    args = parser.parse_args()
    run_benchmark(args.test_dir, args.out_dir)
