# splitsnap/explain.py
"""explain.py - deterministic, traceable plain-language explanations (R5 / R6)."""
from fractions import Fraction
from .split_engine import fmt, TAX_TYPES


def rupee(p):
    return f"-₹{fmt(abs(p))}" if p < 0 else f"₹{fmt(p)}"


def charge_summary(bill):
    """R5 display: Items / Taxes / Service / Packaging / Discount / Rounding / Total."""
    from .split_engine import to_paise
    s = {"Items": sum(to_paise(i["total"]) for i in bill["items"]), "Taxes": 0, "Service Charge": 0,
         "Packaging": 0, "Discount": 0, "Rounding": 0, "Other": 0}
    for c in bill["charges"]:
        a, t = to_paise(c["amount"]), c["type"]
        key = ("Taxes" if t in TAX_TYPES else "Service Charge" if t == "SERVICE_CHARGE" else
               "Packaging" if t == "PACKAGING" else "Discount" if t == "DISCOUNT" else
               "Rounding" if t == "ROUND_OFF" else "Other")
        s[key] += a
    s["Total"] = sum(s.values())
    return {k: v for k, v in s.items() if v != 0 or k in ("Items", "Total")}


def _share(units, total_units):
    return "full" if units == total_units else f"{Fraction(units, total_units)} share"


def _label(t):
    return t if t in TAX_TYPES else t.replace("_", " ").title()


def explain_person(name, pr, items_total=None):
    item_txt = " + ".join(f"{l['name']} ({_share(l['units'], l['total_units'])}, {rupee(l['amount'])})"
                          for l in pr["items"]) or "no items"
    txt = f"{name}: {item_txt} = {rupee(pr['pre_tax'])} pre-tax subtotal."
    if pr["charges"] and any(c["amount"] for c in pr["charges"]):
        parts = ", ".join(f"{_label(c['type'])} {rupee(c['amount'])}" for c in pr["charges"] if c["amount"])
        pct = (pr["charge_total"] / pr["pre_tax"] * 100) if pr["pre_tax"] else 0
        share = ""
        if items_total:                                         # the reason, in numbers: this person's slice of the pre-tax bill
            frac = pr["pre_tax"] / items_total * 100
            share = (f" {name}'s pre-tax subtotal is {frac:.1f}% of the {rupee(items_total)} of items on the bill, "
                     f"so {name} carries {frac:.1f}% of each tax, service charge, discount and rounding line.")
        txt += (f"{share} Charges and discounts are shared in proportion to each person's pre-tax subtotal, not equally: "
                f"{parts}; net {rupee(pr['charge_total'])} ({pct:+.1f}% of {name}'s subtotal).")
    if pr["manual"]:
        txt += (f" Computed total: {rupee(pr['computed_final'])}. MANUAL ADJUSTMENT: set to "
                f"{rupee(pr['final'])} by the user.")
    elif pr.get("auto_rebalanced"):
        txt += (f" Computed total: {rupee(pr['computed_final'])}. Adjusted to {rupee(pr['final'])} "
                f"to rebalance after manual overrides.")
    txt += f" Final: {rupee(pr['final'])}."
    return txt


def explain_all(bill, result):
    lines = ["Bill: " + ", ".join(f"{k} {rupee(v)}" for k, v in charge_summary(bill).items())]
    items_total = sum(pr["pre_tax"] for pr in result["people"].values())
    lines += [explain_person(n, pr, items_total) for n, pr in result["people"].items()]
    if result.get("unreconciled"):
        lines.append(f"Note: manual overrides leave {rupee(result['unreconciled'])} of the bill unassigned.")
    if result["diff_vs_printed"]:
        lines.append(f"Note: items + charges differ from the printed total by {rupee(result['diff_vs_printed'])}.")
    return lines
