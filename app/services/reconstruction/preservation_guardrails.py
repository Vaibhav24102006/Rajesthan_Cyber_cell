import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

FIELD_LABELS = {
    "phone": "Phone Number",
    "transaction_id": "Transaction ID",
    "account_number": "Bank Account",
    "upi_id": "UPI ID",
    "ifsc": "IFSC Code",
    "date": "Date of Event",
    "url": "phishing_url",
    "amount": "Financial Amount",
}


def _entity_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    val = str(value).strip()
    return [val] if val else []


def audit_entity_preservation(
    raw_text: str,
    stage_output: str,
    raw_entities: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reports deterministic entity preservation without mutating the generated output.
    """
    output = stage_output or ""
    total_entities = 0
    preserved_entities = 0
    missing_entities = []

    for key, label in FIELD_LABELS.items():
        for val in _entity_values(raw_entities.get(key)):
            total_entities += 1
            if val in output:
                preserved_entities += 1
            else:
                missing_entities.append({"field": key, "label": label, "value": val})

    if total_entities == 0:
        preservation_rate = 1.0
        status = "no_entities"
    else:
        preservation_rate = preserved_entities / total_entities
        status = "passed" if not missing_entities else "failed"

    return {
        "status": status,
        "total_entities": total_entities,
        "preserved_entities": preserved_entities,
        "missing_entities": missing_entities,
        "preservation_rate": round(preservation_rate, 4),
    }


def verify_and_restore_entities(
    raw_text: str,
    stage_output: str,
    raw_entities: Dict[str, Any]
) -> str:
    """
    Scans the stage_output to verify if all raw deterministic entities are preserved.
    If any entity is lost or altered, it is repaired or appended under a verified telemetry footer.
    """
    if not stage_output:
        return stage_output

    restored_output = stage_output
    audit = audit_entity_preservation(raw_text, stage_output, raw_entities)
    missing_fields = audit.get("missing_entities", [])

    for entity in missing_fields:
        logger.warning(
            "Deterministic entity lost in pass: %s (%s). Restoring...",
            entity["label"],
            entity["value"],
        )

    # Append any missing fields as an investigation telemetry footer to guarantee 100% preservation
    if missing_fields:
        telemetry_lines = ["", "--- Verified Ingestion Telemetry ---"]
        for entity in missing_fields:
            telemetry_lines.append(f"- [Verified {entity['label']}]: {entity['value']}")
        
        # Combine output with telemetry
        restored_output = restored_output.strip() + "\n" + "\n".join(telemetry_lines)

    return restored_output
