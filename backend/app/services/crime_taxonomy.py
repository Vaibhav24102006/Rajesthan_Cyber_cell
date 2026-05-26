from typing import Dict, List, Tuple


TAXONOMY_RULES: List[Tuple[str, List[str]]] = [
    ("Fake Banking App Fraud", ["download", "application", "credit card", "insurance charges", "sbi credit card"]),
    ("OTP Fraud", ["otp", "one time password"]),
    ("UPI Fraud", ["upi", "collect request", "@ok", "@ybl", "@upi"]),
    ("Phishing", ["phishing", "fake link", "whatsapp link", "suspicious link", "http://", "https://", "www."]),
    ("Remote Access Fraud", ["anydesk", "teamviewer", "quicksupport", "remote access"]),
    ("KYC Fraud", ["kyc", "update kyc", "verify kyc"]),
    ("Sextortion", ["sextortion", "private photo", "video leak", "blackmail"]),
    ("APK Scam", ["apk", "install app", "unknown app"]),
]


def classify_by_taxonomy(text: str) -> Dict[str, object]:
    tl = text.lower()
    best_label = "Unknown"
    best_hits = 0
    matched_keywords: List[str] = []

    for label, kws in TAXONOMY_RULES:
        hits = [k for k in kws if k in tl]
        if len(hits) > best_hits:
            best_hits = len(hits)
            best_label = label
            matched_keywords = hits

    confidence = 0.0
    if best_hits >= 3:
        confidence = 0.95
    elif best_hits == 2:
        confidence = 0.8
    elif best_hits == 1:
        confidence = 0.6

    # Severity heuristic for taxonomy engine
    severity = 4
    if best_label in {"OTP Fraud", "UPI Fraud", "Fake Banking App Fraud", "Remote Access Fraud"}:
        severity = 8
    elif best_label in {"Phishing", "KYC Fraud", "APK Scam"}:
        severity = 6
    elif best_label in {"Sextortion"}:
        severity = 7

    return {
        "crime_type_rule": best_label,
        "rule_confidence": confidence,
        "matched_keywords": matched_keywords,
        "rule_severity": severity,
    }

