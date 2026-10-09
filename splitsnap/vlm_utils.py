# splitsnap/vlm_utils.py
"""vlm_utils.py - torch-free helpers for zero-shot VLM baselines: ONE prompt for every model,
robust JSON extraction, and canonicalisation so formatting differences are not scored as errors."""
import json, re
from .schema import CHARGE_TYPES, ensure_schema, normalize_charge_type, parse_money, canon_num

PROMPT = (
    "You are reading a photographed restaurant bill. Return ONLY one JSON object, no markdown, no commentary, "
    "with exactly this schema:\n"
    '{"items":[{"name":str,"qty":str,"unit_price":str,"total":str}],'
    '"charges":[{"type":str,"amount":str}],"subtotal":str,"grand_total":str}\n'
    "Rules: every food/drink line is one item. qty is an integer. unit_price is the price of ONE unit and total is "
    "the line amount. Every tax, service charge, packaging charge, discount or round-off line goes into charges, "
    "never into items. charges.type must be one of: " + ", ".join(CHARGE_TYPES) + ". Discounts are negative numbers. "
    "subtotal is the sum of the items before taxes and charges. grand_total is the final amount payable. "
    "Write numbers as plain decimals without currency symbols or thousands separators."
)


def extract_json(text):
    """First {...} object in the model output (handles ```json fences and chatter). None if unparseable."""
    t = re.sub(r"```(?:json)?", "", str(text))
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(t[a:b + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _num(v):
    d = parse_money(v)
    try:
        return canon_num(d) if d is not None else str(v or "").strip()
    except Exception:                       # absurd magnitudes from garbage output
        return str(v or "").strip()


def normalize_pred(obj):
    """Raw VLM JSON -> target schema with canonical numbers and normalised charge types."""
    e = ensure_schema(obj or {})
    for it in e["items"]:
        it["qty"] = str(int(parse_money(it["qty"]))) if parse_money(it["qty"]) is not None else it["qty"]
        it["unit_price"], it["total"] = _num(it["unit_price"]), _num(it["total"])
    for c in e["charges"]:
        t = c["type"].strip().upper().replace(" ", "_")
        c["type"] = t if t in CHARGE_TYPES else normalize_charge_type(c["type"])
        c["amount"] = _num(c["amount"])
    e["subtotal"], e["grand_total"] = _num(e["subtotal"]), _num(e["grand_total"])
    return e
