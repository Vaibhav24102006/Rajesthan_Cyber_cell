import ipaddress
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+91[\-\s]?)?([6-9]\d{9})(?!\d)")

# Common identifiers used in financial cyber frauds (India)
UPI_PATTERN = re.compile(r"\b([a-zA-Z0-9.\-_]{2,64}@[a-zA-Z]{2,64})\b")
IFSC_PATTERN = re.compile(r"\b([A-Za-z]{4}0[A-Za-z0-9]{6})\b")

URL_PATTERN = re.compile(
    r"\b((?:https?://|www\.)[^\s<>()\"']+)",
    flags=re.IGNORECASE,
)
DOMAIN_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?([^/\s<>()\"']+)", flags=re.IGNORECASE)

IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# Crypto wallet patterns (approximate extraction for intelligence)
ETH_PATTERN = re.compile(r"\b(0x[a-fA-F0-9]{40})\b")
BTC_BECH32_PATTERN = re.compile(r"\b(bc1[0-9a-z]{25,39})\b", flags=re.IGNORECASE)
BTC_LEGACY_PATTERN = re.compile(r"\b([13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
TXN_PATTERN = re.compile(
    r"(?:transaction\s*id|txn\s*id|utr|transaction)\s*[-:]?\s*(\d{10,24})",
    flags=re.IGNORECASE,
)
ACCOUNT_PATTERN = re.compile(
    r"(?:\baccount(?:\s*number)?\b|\ba\/c\b)[\s:\-#]*([0-9]{8,20})",
    flags=re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(
    r"(?:₹|rs\.?|rupees?)\s*([0-9][0-9,]*(?:\.\d{1,2})?)|([0-9][0-9,]*(?:\.\d{1,2})?)\s*(?:₹|rs\.?|rupees?)",
    flags=re.IGNORECASE,
)
DATE_PATTERN_1 = re.compile(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b")
DATE_PATTERN_2 = re.compile(
    r"\b(\d{1,2})\s*(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s*(\d{2,4})\b",
    flags=re.IGNORECASE,
)
TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2})\s*(am|pm)?\b", flags=re.IGNORECASE)


DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


BANKS = _load_json(DATA_DIR / "banks.json", [])
_gaz = _load_json(DATA_DIR / "districts_gazetteer.json", {"rajasthan_districts": [], "other_common_locations": []})
DISTRICTS = list(_gaz.get("rajasthan_districts", [])) + list(_gaz.get("other_common_locations", []))


def _clean_numeric(value: str) -> str:
    return value.replace(",", "").strip()


def _extract_first_match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    if not m:
        return None
    # Prefer capturing group(1) when present; otherwise return full match.
    if m.lastindex and m.lastindex >= 1:
        return m.group(1).strip()
    return m.group(0).strip()


def _normalize_bank_name(value: str) -> str:
    cleaned = " ".join(value.split()).strip()
    upper = cleaned.upper()
    if upper in {"SBI", "HDFC", "ICICI", "PNB", "IDBI", "UCO"}:
        return upper
    return cleaned.title()


def _match_from_whitelist(text: str, candidates: list[str]) -> str | None:
    tl = text.lower()
    best = None
    for c in candidates:
        c_clean = str(c).strip()
        if not c_clean:
            continue
        if c_clean.lower() in tl:
            if best is None or len(c_clean) > len(best):
                best = c_clean
    return best


def _infer_district(text: str) -> str | None:
    return _match_from_whitelist(text, DISTRICTS)


def _extract_domain_from_url(url: str) -> str | None:
    m = DOMAIN_PATTERN.search(url)
    return m.group(1).strip().lower() if m else None


SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".site", ".gq", ".tk", ".ru", ".zip", ".loan", ".work", ".live", ".win"}
SUSPICIOUS_DOMAIN_KEYWORDS = {"login", "verify", "secure", "account", "update", "bank", "support"}


def _is_suspicious_domain(domain: str | None) -> bool:
    if not domain:
        return False
    d = domain.strip().lower()
    if any(d.endswith(tld) for tld in SUSPICIOUS_TLDS):
        return True
    if any(k in d for k in SUSPICIOUS_DOMAIN_KEYWORDS):
        return True
    # Very long / random-looking domains are often phishing.
    if len(d) >= 25 and d.count("-") >= 1:
        return True
    return False


def _is_suspicious_ip(ip: str | None) -> bool:
    if not ip:
        return False
    try:
        obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    # Treat public IPs as more suspicious than private ranges.
    return not (obj.is_private or obj.is_loopback or obj.is_multicast or obj.is_reserved or obj.is_link_local)


def regex_extract(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "victim_name": None,
        "phone": None,
        "location": None,
        "district": None,
        "amount": None,
        "transaction_id": None,
        "account_number": None,
        "bank": None,
        "upi_id": None,
        "ifsc": None,
        "url": None,
        "domain": None,
        "crypto_wallet": None,
        "ip_address": None,
        "suspicious_domain": False,
        "suspicious_ip": False,
        "date": None,
        "time": None,
    }

    phones = PHONE_PATTERN.findall(text)
    if phones:
        result["phone"] = phones[0]

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

    bank = _match_from_whitelist(text, BANKS)
    if bank:
        result["bank"] = _normalize_bank_name(bank)

    district = _infer_district(text)
    if district:
        result["district"] = district
        result["location"] = district

    result["upi_id"] = _extract_first_match(UPI_PATTERN, text)
    result["ifsc"] = _extract_first_match(IFSC_PATTERN, text)

    url = _extract_first_match(URL_PATTERN, text)
    if url:
        url_norm = url.strip()
        if url_norm.lower().startswith("www."):
            url_norm = "https://" + url_norm
        result["url"] = url_norm
        result["domain"] = _extract_domain_from_url(url_norm)
        result["suspicious_domain"] = _is_suspicious_domain(result["domain"])

    wallet = _extract_first_match(ETH_PATTERN, text)
    if not wallet:
        wallet = _extract_first_match(BTC_BECH32_PATTERN, text)
    if not wallet:
        wallet = _extract_first_match(BTC_LEGACY_PATTERN, text)
    result["crypto_wallet"] = wallet

    result["ip_address"] = _extract_first_match(IP_PATTERN, text)
    result["suspicious_ip"] = _is_suspicious_ip(result["ip_address"])

    d1 = DATE_PATTERN_1.search(text)
    if d1:
        dd, mm, yy = d1.group(1), d1.group(2), d1.group(3)
        if len(yy) == 2:
            yy = "20" + yy
        result["date"] = f"{yy.zfill(4)}-{mm.zfill(2)}-{dd.zfill(2)}"
    else:
        d2 = DATE_PATTERN_2.search(text)
        if d2:
            dd, mon, yy = d2.group(1), d2.group(2), d2.group(3)
            if len(yy) == 2:
                yy = "20" + yy
            month_map = {
                "jan": "01",
                "january": "01",
                "feb": "02",
                "february": "02",
                "mar": "03",
                "march": "03",
                "apr": "04",
                "april": "04",
                "may": "05",
                "jun": "06",
                "june": "06",
                "jul": "07",
                "july": "07",
                "aug": "08",
                "august": "08",
                "sep": "09",
                "sept": "09",
                "september": "09",
                "oct": "10",
                "october": "10",
                "nov": "11",
                "november": "11",
                "dec": "12",
                "december": "12",
            }
            mm = month_map.get(mon.lower(), "01")
            result["date"] = f"{yy.zfill(4)}-{mm}-{dd.zfill(2)}"

    tm = TIME_PATTERN.search(text)
    if tm:
        hhmm = tm.group(1)
        ampm = (tm.group(2) or "").lower()
        try:
            hh, mi = hhmm.split(":")
            hh_i = int(hh)
            mi_i = int(mi)
            if ampm == "pm" and 1 <= hh_i < 12:
                hh_i += 12
            if ampm == "am" and hh_i == 12:
                hh_i = 0
            if 0 <= hh_i <= 23 and 0 <= mi_i <= 59:
                result["time"] = f"{hh_i:02d}:{mi_i:02d}"
        except Exception:
            pass

    return result


def deterministic_extract_with_confidence(text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    fields = regex_extract(text)
    conf: Dict[str, float] = {}

    def set_conf(k: str, v: Any, high: float = 0.95, low: float = 0.0):
        conf[k] = high if v not in (None, "", False) else low

    for k in [
        "transaction_id",
        "phone",
        "amount",
        "ifsc",
        "bank",
        "district",
        "date",
        "time",
        "upi_id",
        "account_number",
        "url",
        "domain",
        "crypto_wallet",
        "ip_address",
    ]:
        set_conf(k, fields.get(k))

    conf["suspicious_domain"] = 0.7 if fields.get("suspicious_domain") else 0.0
    conf["suspicious_ip"] = 0.7 if fields.get("suspicious_ip") else 0.0

    return fields, conf
