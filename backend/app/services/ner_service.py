import logging
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


def _safe_load_spacy_model():
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    # Prefer multilingual small model if available; fall back to English.
    for model_name in ("xx_ent_wiki_sm", "en_core_web_sm"):
        try:
            return spacy.load(model_name)
        except Exception:
            continue
    return None


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    _NLP = _safe_load_spacy_model()
    if _NLP is None:
        logger.warning("spaCy model not available; NER enrichment disabled.")
    return _NLP


def ner_extract(text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Optional NER enrichment:
    - victim_name: PERSON candidates
    - location: GPE/LOC candidates (not used for district; district stays gazetteer-only)
    - org: ORG candidates (currently not stored)
    Returns (fields, confidence)
    """
    nlp = _get_nlp()
    if nlp is None:
        return {}, {}

    try:
        doc = nlp(text)
    except Exception:
        return {}, {}

    person = None
    gpe = None
    org = None

    for ent in doc.ents:
        label = getattr(ent, "label_", "")
        val = (ent.text or "").strip()
        if not val:
            continue
        if person is None and label in {"PERSON"} and len(val) >= 3:
            person = val
        if gpe is None and label in {"GPE", "LOC"} and len(val) >= 3:
            gpe = val
        if org is None and label in {"ORG"} and len(val) >= 3:
            org = val

    fields: Dict[str, Any] = {}
    conf: Dict[str, float] = {}

    if person:
        fields["victim_name"] = person
        conf["victim_name"] = 0.6
    if gpe:
        fields["ner_location"] = gpe
        conf["ner_location"] = 0.55
    if org:
        fields["ner_org"] = org
        conf["ner_org"] = 0.55

    return fields, conf

