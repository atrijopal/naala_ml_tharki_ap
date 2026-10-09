# splitsnap/schema.py
"""schema.py - canonical schema, number parsing, charge-label normalisation.
Every number in every ground-truth label (CORD, synthetic, real, SROIE) goes
through canon_num() so the model sees ONE consistent number format."""
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

TASK_START = "<s_splitsnap>"          # decoder start token for the main task
TASK_SROIE = "<s_sroie>"              # optional auxiliary task (company/date/address/total)
CHARGE_TYPES = ["CGST", "SGST", "IGST", "GST", "VAT", "TAX", "SERVICE_CHARGE",
                "PACKAGING", "DISCOUNT", "ROUND_OFF", "OTHER"]
SCHEMA_KEYS = ["items", "name", "qty", "unit_price", "total", "charges", "type",
               "amount", "subtotal", "grand_total"]
SROIE_KEYS = ["company", "date", "address", "total"]


def parse_money(s):
    """'1,250.50' / 'Rs 45.000' / '(12)' / '-5,000' -> Decimal, or None.
    Ambiguity: a lone separator followed by exactly 3 digits is treated as a
    thousands separator ('45.000' -> 45000). Check this against your own bills."""
    if s is None:
        return None
    if isinstance(s, (int, float, Decimal)):
        return Decimal(str(s))
    s = str(s).strip()
    if not s:
        return None
    neg = s.startswith("-") or s.endswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^\d.,]", "", s).strip(".,")     # 'Rs. 1,000.50' leaves a stray leading '.'
    if not s:
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[-1]) == 3:      # thousands separators
            s = "".join(parts)
        else:                                          # decimal separator
            s = parts[0] + "." + parts[1]
    try:
        v = Decimal(s)
    except InvalidOperation:
        return None
    return -v if neg else v


def canon_num(x):
    """Decimal/str/number -> canonical string: 2dp max, trailing zeros stripped."""
    d = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def parse_int(s, default=None):
    m = re.search(r"\d+", str(s)) if s is not None else None
    return int(m.group()) if m else default


CHARGE_RULES = [
    (r"\bcgst\b|central\s*gst", "CGST"),
    (r"\bsgst\b|\butgst\b|state\s*gst", "SGST"),
    (r"\bigst\b", "IGST"),
    (r"service|svc|\bs\s*(charge|chg)", "SERVICE_CHARGE"),
    (r"packag|parcel|\bpkg\b|container", "PACKAGING"),
    (r"discount|\bdisc\b|\boffer\b|coupon|promo", "DISCOUNT"),
    (r"round|\br/?off\b", "ROUND_OFF"),
    (r"\bvat\b", "VAT"),
    (r"\bgst\b", "GST"),
    (r"tax|pb1|ppn", "TAX"),
]


def normalize_charge_type(raw):
    s = str(raw).lower().replace(".", "")
    for pat, typ in CHARGE_RULES:
        if re.search(pat, s):
            return typ
    return "OTHER"


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def flatten(obj, prefix=""):
    """JSON -> list of (path, value). List indices are dropped (as in Donut's F1)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            out += flatten(v, prefix)
    else:
        out.append((prefix, str(obj)))
    return out


def ensure_list(x):
    if x is None or x == "":
        return []
    return x if isinstance(x, list) else [x]      # token2json collapses 1-item lists to dicts!


def ensure_schema(pred):
    """Force a parsed model output into the exact schema (all leaves are strings). Early or garbled model output can parse
    to a list or a string instead of a dict (found in training: validation crashed at epoch 2): treat it as empty."""
    if not isinstance(pred, dict):
        pred = {}
    items = []
    for it in ensure_list(pred.get("items")):
        if isinstance(it, dict):
            items.append({k: str(it.get(k, "")).strip() for k in ("name", "qty", "unit_price", "total")})
    charges = []
    for c in ensure_list(pred.get("charges")):
        if isinstance(c, dict):
            charges.append({"type": str(c.get("type", "OTHER")).strip(),
                            "amount": str(c.get("amount", "")).strip()})
    return {"items": items, "charges": charges,
            "subtotal": str(pred.get("subtotal", "")).strip(),
            "grand_total": str(pred.get("grand_total", "")).strip()}


def make_schema(items, charges, subtotal, grand_total):
    """items: [(name, qty, unit, total)], charges: [(TYPE, amount)] -> canonical dict."""
    return {
        "items": [{"name": n, "qty": str(int(q)), "unit_price": canon_num(u), "total": canon_num(t)}
                  for n, q, u, t in items],
        "charges": [{"type": normalize_charge_type(t) if t not in CHARGE_TYPES else t,
                     "amount": canon_num(a)} for t, a in charges],
        "subtotal": canon_num(subtotal),
        "grand_total": canon_num(grand_total),
    }


PS_CHARGE_NAMES = {"SERVICE_CHARGE": "Service Charge", "ROUND_OFF": "Round Off", "PACKAGING": "Packaging",
                   "DISCOUNT": "Discount", "OTHER": "Other"}      # CGST/SGST/IGST/GST/VAT/TAX keep their codes


def to_ps_schema(d):
    """Our internal string schema -> the problem statement's JSON (requirement R1): numeric qty/prices/amounts and
    readable charge names ("Service Charge", "Discount"). Unparseable numbers become None, never a guess."""
    def num(x):
        v = parse_money(x)
        if v is None:
            return None
        return int(v) if v == v.to_integral_value() else float(v)
    return {
        "items": [{"name": i["name"], "qty": parse_int(i["qty"], 1), "unit_price": num(i["unit_price"]),
                   "total": num(i["total"])} for i in d["items"]],
        "charges": [{"type": PS_CHARGE_NAMES.get(c["type"], c["type"]), "amount": num(c["amount"])} for c in d["charges"]],
        "subtotal": num(d["subtotal"]), "grand_total": num(d["grand_total"]),
    }
