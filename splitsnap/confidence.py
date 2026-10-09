# splitsnap/confidence.py
"""confidence.py - per-field confidence from token probabilities + post-hoc calibration (R2 bonus)."""
import re
from collections import Counter
import numpy as np
from .schema import flatten, norm_name, parse_money, canon_num

LIST_KEYS = ("items", "charges")


def field_confidences(tok, gen_ids, probs):
    """Walk the generated token stream; for every leaf <s_key>...</s_key> emit its path
    ('items[2].total'), decoded value and three aggregates (first-token / mean / min probability)."""
    stack, idx, buf, bufp, out = [], {}, [], [], []
    for tid, p in zip(gen_ids, probs):
        t = tok.convert_ids_to_tokens(int(tid))
        if t in (tok.eos_token, tok.pad_token):
            break
        if t == "<sep/>":
            if stack:
                idx[stack[-1]] = idx.get(stack[-1], 0) + 1
            buf, bufp = [], []
        elif t.startswith("</s_") and t.endswith(">"):
            if buf:
                path = ".".join(f"{k}[{idx.get(k, 0)}]" if k in LIST_KEYS else k for k in stack)
                text = tok.convert_tokens_to_string(buf).strip()
                if text.startswith("<") and text.endswith("/>"):
                    text = text[1:-2]
                out.append({"path": path, "value": text, "p_first": float(bufp[0]),
                            "p_mean": float(np.mean(bufp)), "p_min": float(np.min(bufp))})
            if stack:
                stack.pop()
            buf, bufp = [], []
        elif t.startswith("<s_") and t.endswith(">"):
            stack.append(t[3:-1]); buf, bufp = [], []
        else:
            buf.append(t); bufp.append(p)
    return out


def _nv(path, val):
    return norm_name(val) if path.endswith("name") else (canon_num(parse_money(val)) if parse_money(val) is not None else val)


def label_fields(fields, gt):
    """correct[i] = does this predicted (path,value) exist in the ground truth? (multiset match)."""
    cg = Counter((p, _nv(p, v)) for p, v in flatten(gt))
    res = []
    for f in fields:
        key = (re.sub(r"\[\d+\]", "", f["path"]), _nv(f["path"], f["value"]))
        ok = cg[key] > 0
        if ok:
            cg[key] -= 1
        res.append(int(ok))
    return res


def fit_calibrator(scores, correct):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, correct)


def reliability_table(scores, correct, bins=10):
    s, c = np.asarray(scores, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) | (hi == 1.0) & (s <= hi))
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_conf": float(s[m].mean()), "accuracy": float(c[m].mean())})
    return rows


def ece(scores, correct, bins=10):
    rows, n = reliability_table(scores, correct, bins), len(scores)
    return sum(r["n"] / n * abs(r["mean_conf"] - r["accuracy"]) for r in rows)


def auroc(scores, correct):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(correct, scores)        # how well confidence separates right from wrong fields


def pick_threshold(scores, correct, target_precision=0.98):
    """Lowest score such that accepting everything >= it keeps precision >= target; the rest is flagged."""
    s, c = np.asarray(scores, float), np.asarray(correct, float)
    o = np.argsort(-s); s, c = s[o], c[o]
    prec = np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(prec >= target_precision)[0]
    return float(s[ok[-1]]) if len(ok) else 1.01
