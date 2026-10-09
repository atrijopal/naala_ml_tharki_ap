# splitsnap/tokens.py
"""tokens.py - JSON <-> Donut tagged-token serialisation (same scheme as the official repo)."""
from .schema import SCHEMA_KEYS, SROIE_KEYS, CHARGE_TYPES, TASK_START, TASK_SROIE


def collect_added_tokens():
    added = {"<sep/>", TASK_START, TASK_SROIE}
    for k in SCHEMA_KEYS + SROIE_KEYS:
        added |= {f"<s_{k}>", f"</s_{k}>"}
    for t in CHARGE_TYPES:                    # categorical values -> ONE token each
        added.add(f"<{t}/>")
    return sorted(added)


def json2token(obj, added=None):
    """dict -> '<s_key>value</s_key>...'; list -> items joined by '<sep/>'."""
    if added is None:
        added = set(collect_added_tokens())
    if isinstance(obj, dict):
        return "".join(f"<s_{k}>{json2token(v, added)}</s_{k}>" for k, v in obj.items())
    if isinstance(obj, list):
        return "<sep/>".join(json2token(v, added) for v in obj)
    s = str(obj)
    return f"<{s}/>" if f"<{s}/>" in added else s
