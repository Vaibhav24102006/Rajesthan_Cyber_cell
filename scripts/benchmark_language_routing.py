import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("LANGUAGE_ROUTING_ENABLE_LLM", "false")

from app.services.reconstruction import reconstruct_ocr_text_staged


def main() -> int:
    cases_path = ROOT / "tests" / "language_routing_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    run_dir = ROOT / "storage" / "datasets" / "language_routing" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    failed = False

    for case in cases:
        start = perf_counter()
        final_text, rejected, reason, meta = reconstruct_ocr_text_staged(
            case["raw_ocr_output"],
            ocr_confidence=0.90,
            reconstruction_mode="structured_english_output",
            document_domain="cybercrime_complaint",
        )
        runtime_ms = round((perf_counter() - start) * 1000, 2)
        failure_reasons = []

        if meta.get("detected_language") != case["expected_language"]:
            failure_reasons.append("language_misclassification")
        if case["expected_dialect_contains"] not in (meta.get("detected_dialect") or ""):
            failure_reasons.append("dialect_misclassification")
        if meta.get("routing_decision") != case["expected_routing_decision"]:
            failure_reasons.append("routing_misclassification")
        for token in case["must_preserve"]:
            if token not in final_text:
                failure_reasons.append(f"entity_loss:{token}")
        if meta.get("benchmarks", {}).get("hallucination_rate", 0.0) > 0.05:
            failure_reasons.append("hallucination_threshold_exceeded")
        if rejected:
            failure_reasons.append(f"rejected:{reason}")

        failed = failed or bool(failure_reasons)
        results.append(
            {
                "case_id": case["id"],
                "uploaded_image": case.get("uploaded_image"),
                "raw_ocr_output": case["raw_ocr_output"],
                "detected_language": meta.get("detected_language"),
                "detected_dialect": meta.get("detected_dialect"),
                "routing_decision": meta.get("routing_decision"),
                "normalized_regional_text": meta.get("normalized_regional_text"),
                "translated_text": meta.get("translation_output"),
                "structured_final_output": final_text,
                "expected_ground_truth": case["expected_ground_truth"],
                "confidence_metrics": meta.get("stage_confidence", {}),
                "runtime_metrics": {
                    "benchmark_runtime_ms": runtime_ms,
                    "reconstruction_runtime_ms": meta.get("reconstruction_runtime_ms"),
                },
                "entity_preservation_audit": meta.get("entity_preservation_audit", {}),
                "quality_metrics": meta.get("benchmarks", {}),
                "failure_reasons": failure_reasons,
            }
        )

    output = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_enabled": os.getenv("LANGUAGE_ROUTING_ENABLE_LLM"),
        "quality_targets": {
            "entity_preservation": 1.0,
            "hallucination_rate_max": 0.05,
        },
        "results": results,
    }
    output_path = run_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"run_path": str(output_path), "failed": failed}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
