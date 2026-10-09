# splitsnap/convert_cord.py
"""convert_cord.py - CORD-v2 ground truth -> our target schema (+ export to disk).
CORD key names below are from the CORD repo/class list and from memory; run the inspection
step in the guide (section 4) and fix the .get(...) keys if your copy differs."""
import json, os
from decimal import Decimal
from .schema import (parse_money, parse_int, canon_num, make_schema, TASK_START)
from .validate import is_consistent


def _d(x):
    if isinstance(x, list):
        x = x[0] if x else {}
    return x if isinstance(x, dict) else {}


def cord_to_schema(gt_parse):
    menu = gt_parse.get("menu", [])
    menu = [menu] if isinstance(menu, dict) else (menu or [])
    items = []
    for m in menu:
        name = m.get("nm", "")
        name = " ".join(name) if isinstance(name, list) else str(name or "")
        name = name.strip()
        qty = parse_int(m.get("cnt"), 1) or 1
        total, unit = parse_money(m.get("price")), parse_money(m.get("unitprice"))
        if total is None and unit is not None:
            total = unit * qty
        if unit is None and total is not None:
            unit = total / qty
        if not name or total is None:
            continue
        items.append((name, qty, unit, total))
    st, tot = _d(gt_parse.get("sub_total")), _d(gt_parse.get("total"))
    charges = []

    def add(kind, key, force_neg=False):
        v = parse_money(st.get(key))
        if v is not None and v != 0:
            charges.append((kind, -abs(v) if force_neg else v))
    add("TAX", "tax_price")
    add("SERVICE_CHARGE", "service_price")
    add("OTHER", "othersvc_price")
    add("DISCOUNT", "discount_price", force_neg=True)
    sub = parse_money(st.get("subtotal_price"))
    if sub is None:
        sub = sum((t for _, _, _, t in items), Decimal(0))
    grand = parse_money(tot.get("total_price"))
    if grand is None:
        return None
    return make_schema(items, charges, sub, grand)


def is_outlier(schema, max_total=1e8, max_qty=50):
    """Label noise found by the EDA (a 5.58 billion total, qty 202). Dropped from TRAIN only; val/test stay as published."""
    def f(x):
        v = parse_money(x)
        return float(v) if v is not None else 0.0
    if f(schema["grand_total"]) > max_total or f(schema["grand_total"]) <= 0:
        return True
    return any(int(i["qty"]) > max_qty or f(i["total"]) > max_total or f(i["unit_price"]) > max_total for i in schema["items"])


def export_cord(out_dir="data/cord"):
    """Needs internet + `pip install datasets`. Writes images + data/cord/manifest.jsonl."""
    from datasets import load_dataset
    ds = load_dataset("naver-clova-ix/cord-v2")
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    recs, skipped, inconsistent, outliers = [], 0, 0, 0
    for split in ds.keys():                                  # train / validation / test
        sp = "val" if split == "validation" else split
        for i, ex in enumerate(ds[split]):
            schema = cord_to_schema(json.loads(ex["ground_truth"])["gt_parse"])
            if schema is None or not schema["items"]:
                skipped += 1
                continue
            if sp == "train" and is_outlier(schema):
                outliers += 1
                continue
            inconsistent += (not is_consistent(schema))
            path = f"{out_dir}/images/{sp}_{i:04d}.jpg"
            ex["image"].convert("RGB").save(path, quality=95)
            recs.append({"id": f"cord_{sp}_{i:04d}", "image": path, "gt": schema,
                         "task": TASK_START, "source": "cord", "split": sp})
    with open(f"{out_dir}/manifest.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"CORD: {len(recs)} records, {skipped} skipped, {outliers} train outliers dropped, {inconsistent} fail the arithmetic check")


if __name__ == "__main__":
    export_cord()
