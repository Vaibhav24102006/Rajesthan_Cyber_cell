import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import shutil

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, base_dir: Path):
        self.dataset_dir = base_dir / "datasets" / "ocr_validation"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dataset_dir / "dataset.jsonl"
        
    def save_sample(self, raw_img_path: Path, ocr_results: dict, human_verified: bool = False, verified_text: str = None):
        """
        Saves a sample to the dataset.
        Format is compatible with HuggingFace datasets (JSONL).
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            sample_id = f"sample_{timestamp}_{raw_img_path.stem}"
            
            # Copy original image to dataset directory
            img_dest = self.dataset_dir / f"{sample_id}{raw_img_path.suffix}"
            if raw_img_path.exists():
                shutil.copy2(raw_img_path, img_dest)
            
            # Extract basic metrics
            metadata = ocr_results.get("preprocessing_metadata", {})
            failure_reasons = metadata.get("failure_reasons", [])
            heuristics = metadata.get("heuristics", {})
            
            sample_data = {
                "id": sample_id,
                "image_path": str(img_dest.name),
                "timestamp": timestamp,
                "human_verified": human_verified,
                "raw_ocr_text": ocr_results.get("raw_text", ""),
                "verified_text": verified_text if human_verified else "",
                "confidence": ocr_results.get("confidence", 0.0),
                "failure_reasons": failure_reasons,
                "heuristics": heuristics,
                "engine_usage": metadata.get("engine_usage", {}),
                "line_results": ocr_results.get("line_results", [])
            }
            
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                
            logger.info(f"Saved dataset sample: {sample_id}")
            return sample_id
            
        except Exception as e:
            logger.error(f"Failed to save dataset sample: {e}")
            return None

def get_dataset_manager():
    # Base dir is two levels up from this file (app level -> backend level -> storage level)
    # Actually, let's use the standard storage dir
    storage_dir = Path(__file__).resolve().parents[4] / "storage"
    return DatasetManager(storage_dir)
