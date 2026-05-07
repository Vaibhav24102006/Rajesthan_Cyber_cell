import re
from typing import Any, Dict


PHONE_PATTERN = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}")
TXN_PATTERN = re.compile(
    r"(?:txn|transaction(?:\s*id)?|utr)?[\s:\-#]*([A-Za-z0-9]{10,24}|\d{10,24})",
    flags=re.IGNORECASE,
)
ACCOUNT_PATTERN = re.compile(
    r"(?:account(?:\s*number)?|a\/c)?[\s:\-#]*([0-9]{8,20})",
    flags=re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(
    r"(?:₹|rs\.?|rupees?)\s*([0-9][0-9,]*(?:\.\d{1,2})?)|([0-9][0-9,]*(?:\.\d{1,2})?)\s*(?:₹|rs\.?|rupees?)",
    flags=re.IGNORECASE,
)
BANK_PATTERN = re.compile(
    r"(?:bank|financial institution|company)\s*[:\-]?\s*([A-Za-z ]{3,50})",
    flags=re.IGNORECASE,
)
LOCATION_PATTERN = re.compile(
    r"(?:in|at|from|location|address)\s*[:\-]?\s*([A-Za-z0-9\-\s,]{4,100})",
    flags=re.IGNORECASE,
)


def _clean_numeric(value: str) -> str:
    return value.replace(",", "").strip()


def regex_extract(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "victim_name": None,
        "phone": None,
        "location": None,
        "amount": None,
        "transaction_id": None,
        "account_number": None,
        "bank": None,
    }

    phone_match = PHONE_PATTERN.search(text)
    if phone_match:
        phone = re.sub(r"\D", "", phone_match.group(0))
        result["phone"] = phone[-10:]

    txn_match = TXN_PATTERN.search(text)
    if txn_match:
        result["transaction_id"] = txn_match.group(1).strip()

    acc_match = ACCOUNT_PATTERN.search(text)
    if acc_match:
        result["account_number"] = acc_match.group(1).strip()

    amount_match = AMOUNT_PATTERN.search(text)
    if amount_match:
        amount_raw = amount_match.group(1) or amount_match.group(2)
        result["amount"] = _clean_numeric(amount_raw)

    bank_match = BANK_PATTERN.search(text)
    if bank_match:
        result["bank"] = " ".join(bank_match.group(1).split()).title()

    loc_match = LOCATION_PATTERN.search(text)
    if loc_match:
        result["location"] = " ".join(loc_match.group(1).split()).strip(", ")

    return result
