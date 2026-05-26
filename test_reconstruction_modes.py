import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure backend root is in import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.indus_reconstruction_service import (
    reconstruct_ocr_text,
    detect_semantic_domain,
)
from app.services.ocr_correction import correct_ocr_text


class TestReconstructionSystem(unittest.TestCase):

    def test_domain_heuristic_classifier(self):
        """Verify that detect_semantic_domain correctly categorizes text based on keywords."""
        cyber_text = "I received a link on WhatsApp and lost money in a UPI transaction fraud"
        bank_text = "My account passbook and cheque ATM debit card transfer has issue at branch"
        legal_text = "Shriman sho prarthna patra police complaint station theft"
        edu_text = "Mauryan architecture sanchi stupas ashoka emperor history notes"
        general_text = "hello world some random handwriting text notes"

        self.assertEqual(detect_semantic_domain(cyber_text), "cybercrime_complaint")
        self.assertEqual(detect_semantic_domain(bank_text), "banking_fraud")
        self.assertEqual(detect_semantic_domain(legal_text), "legal_complaint")
        self.assertEqual(detect_semantic_domain(edu_text), "educational_content")
        self.assertEqual(detect_semantic_domain(general_text), "notes_documentation")

    @patch("requests.post")
    def test_local_correction_prompt_building(self, mock_post):
        """Verify that correct_ocr_text correctly formats advanced prompt with mode, domain, and fields."""
        # Set up a mock response from Ollama
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"response": "Normalized sentence: Mauryan architecture at Sanchi stupas."}
        mock_post.return_value = mock_response

        raw_ocr = "Mauyn archttue at Sanchi stupas. Transaction ID: 123456789012. Call 9876543210."
        
        # Test Balanced Reconstruction + Educational Notes with moderate confidence
        text, was_rejected, reason, meta = correct_ocr_text(
            raw_text=raw_ocr,
            ocr_confidence=0.75,
            reconstruction_mode="balanced_reconstruction",
            document_domain="educational_content"
        )

        # Assert correct service response was returned
        self.assertFalse(was_rejected)
        self.assertEqual(text, "Normalized sentence: Mauryan architecture at Sanchi stupas.")
        self.assertEqual(meta["detected_domain"], "educational_content")
        self.assertEqual(meta["reconstruction_mode"], "balanced_reconstruction")

        # Verify the payload sent to Ollama
        self.assertTrue(mock_post.called)
        payload = mock_post.call_args[1]["json"]
        prompt_sent = payload["prompt"]

        # Assert prompt contains domain info
        self.assertIn("Educational / Academic Notes", prompt_sent)
        self.assertIn("Mauryan", prompt_sent)
        
        # Assert prompt contains mode description
        self.assertIn("RECONSTRUCTION MODE: Balanced Reconstruction", prompt_sent)
        
        # Assert prompt contains confidence instructions
        self.assertIn("CONFIDENCE AWARENESS", prompt_sent)
        self.assertIn("moderate", prompt_sent.lower())

        # Assert deterministic fields are extracted and listed
        self.assertIn("DETERMINISTIC FIELDS", prompt_sent)
        self.assertIn("9876543210", prompt_sent)
        self.assertIn("123456789012", prompt_sent)

    @patch("requests.post")
    def test_reconstruction_modes_indus_fallback(self, mock_post):
        """Verify that reconstruct_ocr_text properly falls back to local when Indus is unconfigured."""
        # If INDUS_API_URL is empty, it should call correct_ocr_text
        # We will mock the Ollama POST call inside correct_ocr_text
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"response": "Mauryan architecture at Sanchi stupas."}
        mock_post.return_value = mock_response

        raw_ocr = "Mauyn archttue at Sanchi stupas."

        # Make sure INDUS_API_URL is unconfigured in service module
        with patch("app.services.indus_reconstruction_service.INDUS_API_URL", ""):
            text, was_rejected, reason, meta = reconstruct_ocr_text(
                raw_text=raw_ocr,
                ocr_confidence=0.60, # Low confidence
                reconstruction_mode="aggressive_semantic_repair",
                document_domain="auto" # Trigger auto-detect
            )

            # Auto domain-detection should detect educational notes
            self.assertEqual(meta["detected_domain"], "educational_content")
            self.assertEqual(meta["reconstruction_mode"], "aggressive_semantic_repair")
            self.assertEqual(meta["reconstruction_source"], "local_correction_fallback")
            self.assertEqual(text, "Mauryan architecture at Sanchi stupas.")

            # Let's inspect the prompt payload to check aggressive mode and low confidence instructions
            payload = mock_post.call_args[1]["json"]
            prompt_sent = payload["prompt"]
            
            self.assertIn("RECONSTRUCTION MODE: Aggressive Semantic Repair", prompt_sent)
            self.assertIn("OCR Confidence: 0.60 - LOW", prompt_sent)


if __name__ == "__main__":
    unittest.main()
