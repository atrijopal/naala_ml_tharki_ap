# splitsnap/split_engine.py
"""split_engine.py - exact bill splitting in integer paise (no floats, no rounding drift)."""
import copy
from decimal import Decimal, ROUND_HALF_UP
from .schema import parse_money

TAX_TYPES = {"CGST", "SGST", "IGST", "GST", "VAT", "TAX"}


def to_paise(x):
    d = parse_money(x)
    return 0 if d is None else int((d * 100).to_integral_value(rounding=ROUND_HALF_UP))


def fmt(p):
    sign = "-" if p < 0 else ""
    p = abs(p)
    return f"{sign}{p // 100}.{p % 100:02d}"


def largest_remainder(total, weights):
    """Split an integer `total` (paise, may be negative) proportionally to integer `weights`
    so the parts sum EXACTLY to total. Ties go to the lower index (deterministic)."""
    n = len(weights)
    if n == 0:
        return []
    if total == 0:
        return [0] * n
    if sum(weights) <= 0:                      # nobody has a pre-tax share -> split equally
        weights = [1] * n
    sw = sum(weights)
    sign = 1 if total > 0 else -1
    t = abs(total)
    base = [(t * w) // sw for w in weights]
    rem = [(t * w) % sw for w in weights]
    left = t - sum(base)
    for i in sorted(range(n), key=lambda i: (-rem[i], i))[:left]:
        base[i] += 1
    return [sign * b for b in base]


def split_bill(bill, people, assignments):
    """bill: schema dict (items/charges). people: list of names.
    assignments: {item_index: {person: share_units}}  e.g. {1: {"Riya": 1, "Aman": 1, "Sara": 1}}.
    Items are split by share units; every non-item charge (tax, service, packaging, discount,
    rounding) is split in proportion to each person's PRE-TAX subtotal."""
    res = {p: {"items": [], "pre_tax": 0, "charges": [], "charge_total": 0, "final": 0,
               "manual": False} for p in people}
    for i, it in enumerate(bill["items"]):
        w = {k: int(v) for k, v in assignments.get(i, {}).items() if int(v) > 0}
        if not w:
            raise ValueError(f"Item {i} ({it['name']}) is not assigned to anyone")
        unknown = [k for k in w if k not in res]
        if unknown:
            raise ValueError(f"Unknown person(s) {unknown} on item {i}")
        names = list(w)
        parts = largest_remainder(to_paise(it["total"]), [w[k] for k in names])
        tot_units = sum(w.values())
        for k, amt in zip(names, parts):
            res[k]["items"].append({"idx": i, "name": it["name"], "units": w[k],
                                    "total_units": tot_units, "amount": amt})
            res[k]["pre_tax"] += amt
    weights = [res[p]["pre_tax"] for p in people]
    for c in bill["charges"]:
        parts = largest_remainder(to_paise(c["amount"]), weights)
        for p, amt in zip(people, parts):
            res[p]["charges"].append({"type": c["type"], "amount": amt})
            res[p]["charge_total"] += amt
    for p in people:
        res[p]["final"] = res[p]["pre_tax"] + res[p]["charge_total"]
    computed_total = sum(r["final"] for r in res.values())
    printed = to_paise(bill["grand_total"])
    return {"people": res, "computed_total": computed_total, "printed_total": printed,
            "diff_vs_printed": printed - computed_total}


def apply_overrides(result, overrides, rebalance=False):
    """overrides: {person: new_final_paise}. Overridden people are flagged manual=True.
    rebalance=True spreads the leftover (printed_total - sum(finals)) over the NON-overridden
    people in proportion to their computed finals; otherwise the gap is just reported."""
    out = copy.deepcopy(result)
    for p, v in overrides.items():
        out["people"][p]["computed_final"] = out["people"][p]["final"]
        out["people"][p]["final"] = int(v)
        out["people"][p]["manual"] = True
    others = [p for p in out["people"] if p not in overrides]
    gap = out["printed_total"] - sum(r["final"] for r in out["people"].values())
    if rebalance and others and gap:
        parts = largest_remainder(gap, [max(out["people"][p]["final"], 0) for p in others])
        for p, d in zip(others, parts):
            out["people"][p]["computed_final"] = out["people"][p]["final"]
            out["people"][p]["final"] += d
            out["people"][p]["auto_rebalanced"] = True
        gap = 0
    out["unreconciled"] = gap
    return out
