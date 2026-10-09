# splitsnap/validate.py
"""validate.py - arithmetic checks that catch hallucinated / misread numbers."""
from decimal import Decimal
from .schema import parse_money

D0 = Decimal(0)


def suggest_unit_price(qty, total):
    """line total / qty as a plain string, or None when it does not come out as a whole number of paise (then the total is the likelier culprit)."""
    if qty is None or qty <= 0 or total is None:
        return None
    u = total / qty
    if u != u.quantize(Decimal("0.01")):
        return None
    return format(u.quantize(Decimal("0.01")).normalize(), "f")


def repair_unit_prices(d, tol=Decimal("0.50")):
    """A copy of the bill with every suggested rate applied. Used to MEASURE the suggestion (scripts/repair_check.py); the app only offers it."""
    out = {**d, "items": [dict(it) for it in d["items"]]}
    for i, it in enumerate(out["items"]):
        q, u, t = parse_money(it["qty"]), parse_money(it["unit_price"]), parse_money(it["total"])
        if q is not None and u is not None and t is not None and abs(q * u - t) > tol:
            fix = suggest_unit_price(q, t)
            if fix is not None:
                it["unit_price"] = fix
    return out


def check_arithmetic(d, tol=Decimal("0.50")):
    """Returns a list of {path, severity, message}. Paths use the same
    'items[2].total' style as the confidence records so the UI can highlight them."""
    issues = []
    sum_items = D0
    for i, it in enumerate(d["items"]):
        q, u, t = parse_money(it["qty"]), parse_money(it["unit_price"]), parse_money(it["total"])
        if t is None:
            issues.append({"path": f"items[{i}].total", "severity": "error", "message": "missing/unreadable line total"})
            continue
        sum_items += t
        if q is not None and u is not None and abs(q * u - t) > tol:
            issue = {"path": f"items[{i}].total", "severity": "warn", "message": f"qty x unit_price = {q * u} but line total is {t}"}
            fix = suggest_unit_price(q, t)
            if fix is not None:                                  # a misread rate is the commonest error: offer line total / qty as a one-tap fix
                issue["suggestion"] = {"path": f"items[{i}].unit_price", "value": fix, "label": f"Set the rate to {fix}"}
            issues.append(issue)
    sub = parse_money(d["subtotal"])
    if sub is None:
        issues.append({"path": "subtotal", "severity": "warn", "message": "subtotal missing; using sum of items"})
        base = sum_items
    else:
        base = sub
        if abs(sum_items - sub) > tol:
            issues.append({"path": "subtotal", "severity": "error",
                           "message": f"items sum to {sum_items} but subtotal is {sub}"})
    ch_sum = D0
    for i, c in enumerate(d["charges"]):
        a = parse_money(c["amount"])
        if a is None:
            issues.append({"path": f"charges[{i}].amount", "severity": "error", "message": "unreadable charge amount"})
        else:
            ch_sum += a
    gt = parse_money(d["grand_total"])
    if gt is None:
        issues.append({"path": "grand_total", "severity": "error", "message": "grand total missing/unreadable"})
    else:
        diff = gt - (base + ch_sum)
        if abs(diff) > tol:
            has_round = any(c["type"] == "ROUND_OFF" for c in d["charges"])
            hint = " (looks like a missing ROUND_OFF line)" if (abs(diff) < 1 and not has_round) else ""
            issues.append({"path": "grand_total", "severity": "error",
                           "message": f"subtotal + charges = {base + ch_sum} but grand total is {gt}{hint}"})
    return issues


def is_consistent(d):
    return not any(i["severity"] == "error" for i in check_arithmetic(d))
