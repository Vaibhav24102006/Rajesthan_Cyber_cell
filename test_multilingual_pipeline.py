import sys
import os
from pathlib import Path

# Add the backend directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.reconstruction.pipeline import reconstruct_ocr_text_staged

def run_test():
    print("======================================================================")
    print("RUNNING MULTILINGUAL REGIONAL COMPLAINT RECONSTRUCTION VALIDATION TEST")
    print("======================================================================")
    
    # 1. Test case: Mewari/Marwari mixed complaint with amount and account details
    raw_input = "mharo bank account se 50000 kat gayo, bank account number 123456789012, upi pin konya diyo. please help."
    print(f"\n[Test Input - Rajasthani Dialect Complaint]:\n'{raw_input}'\n")
    
    formatted_text, was_rejected, reason, meta = reconstruct_ocr_text_staged(
        raw_text=raw_input,
        ocr_confidence=0.82,
        reconstruction_mode="structured_english_output",
        document_domain="cybercrime_complaint"
    )
    
    print("----------------------------------------------------------------------")
    print(f"Pass 2: Detected Language:   {meta['detected_language']}")
    print(f"Pass 2: Detected Dialect:    {meta['detected_dialect']}")
    print(f"Pass 3: Normalized Hindi:    {meta['normalized_dialect_text']}")
    print(f"Pass 4: Translated English:  {meta['translated_text']}")
    print(f"Pass 5: Reconstructed:       {meta['reconstructed_text']}")
    print(f"Pass 6: Final Structured Output:\n{formatted_text}")
    print("----------------------------------------------------------------------")
    
    # Validation checks
    assert "123456789012" in formatted_text, "Account number should be preserved"
    assert "50000" in formatted_text or "50000" in meta['normalized_dialect_text'], "Amount 50000 should be preserved"
    
    benchmarks = meta["benchmarks"]
    print(f"Word Recovery Rate (WRR):            {benchmarks['word_recovery_rate'] * 100:.1f}%")
    print(f"Readability Improvement (RI):        {benchmarks['readability_improvement'] * 100:.1f}%")
    print(f"Hallucination Rate (HR):             {benchmarks['hallucination_rate'] * 100:.1f}%")
    print(f"Field Preservation Rate (DFPR):      {benchmarks['field_preservation_rate'] * 100:.1f}%")
    print(f"Runtime:                             {meta['reconstruction_runtime_ms']} ms")
    
    print("\n======================================================================")
    print("ALL MULTILINGUAL REGIONAL RECONSTRUCTION VALIDATION TESTS PASSED successfully!")
    print("======================================================================")

if __name__ == "__main__":
    run_test()
