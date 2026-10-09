# splitsnap/metrics_sroie.py
"""metrics_sroie.py - scoring for the SROIE auxiliary task (company/date/address/total) and the checkpoint-selection
score used by train.py. Torch-free."""
import re
from difflib import SequenceMatcher
from .schema import parse_money, canon_num

FIELDS = ("company", "date", "address", "total")


def _alnum(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _total(s):
    try:
        m = parse_money(s)
        return canon_num(m) if m is not None else ""
    except Exception:
        return ""


def field_match(field, pred, gt):
    """(exact, similarity) after normalisation. total compares numbers; date/company/address compare letters+digits."""
    p, g = str(pred or "").strip(), str(gt or "").strip()
    if field == "total":
        a, b = _total(p), _total(g)
    else:
        a, b = _alnum(p), _alnum(g)
    if not a and not b:
        return True, 1.0
    em = a == b
    return em, 1.0 if em else (SequenceMatcher(None, a, b).ratio() if field in ("company", "address") else 0.0)


def evaluate_sroie(preds, gts):
    n = max(len(gts), 1); em = {f: 0 for f in FIELDS}; sim = {f: 0.0 for f in FIELDS}
    tp = npred = ngt = 0
    for p, g in zip(preds, gts):
        p = p if isinstance(p, dict) else {}
        for f in FIELDS:
            e, s = field_match(f, p.get(f), g.get(f))
            em[f] += e; sim[f] += s
            has_p, has_g = bool(str(p.get(f, "")).strip()), bool(str(g.get(f, "")).strip())
            npred += has_p; ngt += has_g; tp += (e and has_p and has_g)
    out = {"n": len(gts)}
    out.update({f"{f}_em": round(em[f] / n, 4) for f in FIELDS})
    out.update({"company_sim": round(sim["company"] / n, 4), "address_sim": round(sim["address"] / n, 4)})
    pr, rc = tp / npred if npred else 0.0, tp / ngt if ngt else 0.0
    out.update({"field_precision": round(pr, 4), "field_recall": round(rc, 4),
                "field_f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0})
    out["score"] = round((out["company_sim"] + out["date_em"] + out["address_sim"] + out["total_em"]) / 4, 4)
    return out


def selection_score(res):
    """Checkpoint-selection score = mean of (CORD TED, SROIE score, real-bill TED) over the groups present: the PS grades
    held-out CORD/SROIE AND real bills. Synthetic is excluded (circular); falls back to the pooled TED if none is present."""
    parts = []
    if "cord" in res:
        parts.append(res["cord"]["ted_acc"])
    if "sroie" in res:
        parts.append(res["sroie"]["score"])
    if "real" in res:
        parts.append(res["real"]["ted_acc"])
    if not parts:
        parts = [res["all"]["ted_acc"]] if "all" in res else [0.0]
    return sum(parts) / len(parts)
