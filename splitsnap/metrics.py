# splitsnap/metrics.py
"""metrics.py - field-level F1, tree-edit-distance accuracy (TED), item/charge F1, totals accuracy.
NOTE: this is our own simplified implementation. For headline numbers on CORD you may also
run the official clovaai/donut test.py, and say in the report which one you used."""
from collections import Counter
from decimal import Decimal
from difflib import SequenceMatcher
import zss
from nltk import edit_distance
from .schema import flatten, norm_name, parse_money, ensure_schema
from .validate import is_consistent


def to_tree(obj, label="root"):
    n = zss.Node(label)
    if isinstance(obj, dict):
        for k in sorted(obj):
            n.addkid(to_tree(obj[k], k))
    elif isinstance(obj, list):
        for v in obj:
            n.addkid(to_tree(v, "[]"))
    else:
        n.addkid(zss.Node(str(obj)))
    return n


def _upd(a, b):
    la, lb = a.label, b.label
    return edit_distance(la, lb) / max(len(la), len(lb), 1)


def _ted(a, b):
    return zss.distance(a, b, zss.Node.get_children, lambda n: 1, lambda n: 1, _upd)


def ted_accuracy(pred, gt):
    d = _ted(to_tree(pred), to_tree(gt))
    d_empty = _ted(zss.Node("root"), to_tree(gt))
    return max(0.0, 1.0 - d / max(d_empty, 1e-9))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def field_f1(preds, gts):
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        cp, cg = Counter(flatten(p)), Counter(flatten(g))
        inter = sum((cp & cg).values())
        tp += inter
        fp += sum(cp.values()) - inter
        fn += sum(cg.values()) - inter
    return _prf(tp, fp, fn)


def _eq(a, b, tol=Decimal("0.005")):
    return a is not None and b is not None and abs(a - b) <= tol


def match_items(pred_items, gt_items, thr=0.8):
    used, tp = set(), 0
    for g in gt_items:
        best, best_r = None, 0.0
        for j, p in enumerate(pred_items):
            if j in used:
                continue
            r = SequenceMatcher(None, norm_name(p["name"]), norm_name(g["name"])).ratio()
            if (r >= thr and r > best_r and _eq(parse_money(p["qty"]), parse_money(g["qty"]))
                    and _eq(parse_money(p["total"]), parse_money(g["total"]))):
                best, best_r = j, r
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(pred_items) - tp, len(gt_items) - tp


def evaluate_records(preds, gts):
    """preds/gts: lists of dicts in the target schema (preds may be raw; they are normalised)."""
    preds = [ensure_schema(p) for p in preds]
    n = max(len(gts), 1)
    ted = sum(ted_accuracy(p, g) for p, g in zip(preds, gts)) / n
    fp_, fr_, ff_ = field_f1(preds, gts)
    itp = ifp = ifn = ctp = cfp = cfn = 0
    total_ok = sub_ok = 0
    for p, g in zip(preds, gts):
        a, b, c = match_items(p["items"], g["items"])
        itp, ifp, ifn = itp + a, ifp + b, ifn + c
        cp = Counter((x["type"], str(parse_money(x["amount"]))) for x in p["charges"])
        cg = Counter((x["type"], str(parse_money(x["amount"]))) for x in g["charges"])
        inter = sum((cp & cg).values())
        ctp, cfp, cfn = ctp + inter, cfp + sum(cp.values()) - inter, cfn + sum(cg.values()) - inter
        total_ok += _eq(parse_money(p["grand_total"]), parse_money(g["grand_total"]))
        sub_ok += _eq(parse_money(p["subtotal"]), parse_money(g["subtotal"]))
    return {
        "n": len(gts),
        "ted_acc": round(float(ted), 4),
        "field_precision": round(fp_, 4), "field_recall": round(fr_, 4), "field_f1": round(ff_, 4),
        "item_f1": round(_prf(itp, ifp, ifn)[2], 4),
        "charge_f1": round(_prf(ctp, cfp, cfn)[2], 4),
        "grand_total_acc": round(total_ok / n, 4),
        "subtotal_acc": round(sub_ok / n, 4),
        "pred_arith_consistent": round(sum(is_consistent(p) for p in preds) / n, 4),
    }
