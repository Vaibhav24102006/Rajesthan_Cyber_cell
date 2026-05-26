import re
from typing import Any, Dict, List, Tuple


PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+91[\-\s]?)?([6-9]\d{9})(?!\d)")
WHATSAPP_LABEL_PATTERN = re.compile(r"(?:whatsapp(?:\s*number)?|whats\s*app(?:\s*number)?)\s*[:\-]?\s*(?:\+91[\-\s]?)?([6-9]\d{9})", flags=re.IGNORECASE)
FRAUD_CALL_LABEL_PATTERN = re.compile(r"(?:fraud\s*call\s*number|call(?:er)?\s*number|mobile\s*number)\s*[:\-]?\s*(?:\+91[\-\s]?)?([6-9]\d{9})", flags=re.IGNORECASE)
URL_PATTERN = re.compile(r"\b((?:https?://|www\.)[^\s<>()\"']+)", flags=re.IGNORECASE)
UPI_PATTERN = re.compile(r"\b([a-zA-Z0-9.\-_]{2,64}@[a-zA-Z]{2,64})\b")
IFSC_PATTERN = re.compile(r"\b([A-Za-z]{4}0[A-Za-z0-9]{6})\b")

TRANSACTION_LABEL_PATTERN = re.compile(
    r"(?:transaction\s*id|txn\s*id|utr|transaction)\s*[-:]?\s*(\d{10,24})",
    flags=re.IGNORECASE,
)
ACCOUNT_LABEL_PATTERN = re.compile(
    r"(?:bank\s*account\s*number|account\s*number|transfer\s*account|ac(?:count)?(?:_holder)?(?:\s*number)?|a\/c)\s*[-:]?\s*(\d{8,20})",
    flags=re.IGNORECASE,
)
AMOUNT_PATTERNS = [
    re.compile(r"amount\s*deducted\s*[:=-]?\s*([0-9][0-9,]*)", flags=re.IGNORECASE),
    re.compile(r"amount\s*lost\s*[:=-]?\s*([0-9][0-9,]*)", flags=re.IGNORECASE),
    re.compile(r"(?:₹|rs\.?|rupees?)\s*([0-9][0-9,]*)", flags=re.IGNORECASE),
    re.compile(r"([0-9][0-9,]*)\s*(?:₹|rs\.?|rupees?)", flags=re.IGNORECASE),
]

DATE_PATTERN = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")
TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2})\s*(am|pm)?\b", flags=re.IGNORECASE)


def _clean_num(v: str) -> str:
    return v.replace(",", "").strip()


def _first_amount(text: str) -> str | None:
    for p in AMOUNT_PATTERNS:
        m = p.search(text)
        if m:
            return _clean_num(m.group(1))
    return None


def _extract_date(text: str) -> str | None:
    m = DATE_PATTERN.search(text)
    if not m:
        return None
    dd, mm, yy = m.group(1), m.group(2), m.group(3)
    if len(yy) == 2:
        yy = "20" + yy
    return f"{yy.zfill(4)}-{mm.zfill(2)}-{dd.zfill(2)}"


def _extract_time(text: str) -> str | None:
    m = TIME_PATTERN.search(text)
    if not m:
        return None
    hhmm = m.group(1)
    ampm = (m.group(2) or "").lower()
    try:
        hh, mi = hhmm.split(":")
        hh_i = int(hh)
        mi_i = int(mi)
        if ampm == "pm" and 1 <= hh_i < 12:
            hh_i += 12
        if ampm == "am" and hh_i == 12:
            hh_i = 0
        if 0 <= hh_i <= 23 and 0 <= mi_i <= 59:
            return f"{hh_i:02d}:{mi_i:02d}"
    except Exception:
        return None
    return None


def parse_cyber_entities(text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    tl = text.lower()

    phones: List[Dict[str, str]] = []
    seen_phones = set()

    # 1) Explicit labeled extraction first
    for m in FRAUD_CALL_LABEL_PATTERN.finditer(text):
        num = m.group(1)
        if num not in seen_phones:
            seen_phones.add(num)
            phones.append({"number": num, "type": "suspect_phone"})

    for m in WHATSAPP_LABEL_PATTERN.finditer(text):
        num = m.group(1)
        if num not in seen_phones:
            seen_phones.add(num)
            phones.append({"number": num, "type": "whatsapp_contact"})

    # 2) Fallback generic number extraction
    for m in PHONE_PATTERN.finditer(text):
        num = m.group(1)
        if num in seen_phones:
            continue
        seen_phones.add(num)

        start = max(0, m.start() - 45)
        end = min(len(text), m.end() + 45)
        ctx = text[start:end].lower()
        ptype = "phone"
        if "whatsapp" in ctx:
            ptype = "whatsapp_contact"
        elif "fraud" in ctx or "call" in ctx:
            ptype = "suspect_phone"
        phones.append({"number": num, "type": ptype})

    transactions: List[Dict[str, str]] = []
    seen_txn = set()
    for m in TRANSACTION_LABEL_PATTERN.finditer(text):
        v = m.group(1)
        if v not in seen_txn:
            seen_txn.add(v)
            transactions.append({"id": v, "type": "transaction_id"})

    accounts: List[Dict[str, str]] = []
    seen_acc = set()
    for m in ACCOUNT_LABEL_PATTERN.finditer(text):
        v = m.group(1)
        if v not in seen_acc:
            seen_acc.add(v)
            atype = "account_number"
            local = m.group(0).lower()
            if "transfer" in local:
                atype = "transfer_account"
            accounts.append({"number": v, "type": atype})

    urls: List[Dict[str, str]] = []
    seen_urls = set()
    for m in URL_PATTERN.finditer(text):
        u = m.group(1).strip()
        if u.lower().startswith("www."):
            u = "https://" + u
        if u not in seen_urls:
            seen_urls.add(u)
            urls.append({"url": u, "type": "suspicious_url"})

    upi_ids = sorted({m.group(1).strip() for m in UPI_PATTERN.finditer(text)})
    ifsc_codes = sorted({m.group(1).strip().upper() for m in IFSC_PATTERN.finditer(text)})

    amount = _first_amount(text)
    date = _extract_date(text)
    time = _extract_time(text)

    primary_phone = phones[0]["number"] if phones else None
    primary_txn = transactions[0]["id"] if transactions else None
    account_primary = next((a for a in accounts if a.get("type") == "account_number"), None)
    primary_acc = account_primary["number"] if account_primary else (accounts[0]["number"] if accounts else None)
    primary_url = urls[0]["url"] if urls else None
    primary_upi = upi_ids[0] if upi_ids else None
    primary_ifsc = ifsc_codes[0] if ifsc_codes else None

    entities: Dict[str, Any] = {
        "phones": phones,
        "accounts": accounts,
        "transactions": transactions,
        "urls": urls,
        "upi_ids": upi_ids,
        "ifsc_codes": ifsc_codes,
        "phone": primary_phone,
        "transaction_id": primary_txn,
        "account_number": primary_acc,
        "url": primary_url,
        "upi_id": primary_upi,
        "ifsc": primary_ifsc,
        "amount": amount,
        "date": date,
        "time": time,
    }

    confidence: Dict[str, float] = {
        "phones": 0.95 if phones else 0.0,
        "accounts": 0.95 if accounts else 0.0,
        "transactions": 0.95 if transactions else 0.0,
        "urls": 0.9 if urls else 0.0,
        "amount": 0.9 if amount else 0.0,
        "date": 0.9 if date else 0.0,
        "time": 0.85 if time else 0.0,
        "upi_id": 0.95 if primary_upi else 0.0,
        "ifsc": 0.95 if primary_ifsc else 0.0,
    }
    return entities, confidence

