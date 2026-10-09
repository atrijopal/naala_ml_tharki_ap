"""Error breakdown on the 100 unseen generated bills: what kinds of mistakes does the fine-tuned model make, and how many does the confidence flag point at?
  python scripts/error_analysis.py --ckpt ckpt/best_fp16 --out results/error_analysis.json"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from splitsnap.infer import generate_json
from splitsnap.model import load_model
from splitsnap.schema import ensure_schema

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--folder", default="examples/final_run"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", default="results/error_analysis.json"); ap.add_argument("--calibration", default="results/calibration.json"); a = ap.parse_args()
THR = json.load(open(a.calibration))["aggregators"]["p_min"]["flag_below"]
num = lambda x: (lambda s: round(float(s), 2) if s.replace(".", "", 1).replace("-", "", 1).isdigit() else None)(str(x).replace(",", "").strip())
nm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
model, processor = load_model(a.ckpt); model.to("cuda").half().eval()
cats = {k: 0 for k in ("item name misread", "quantity misread", "unit price misread", "line total misread", "item missed", "item added", "charge type wrong", "charge amount wrong", "charge missed", "charge added", "subtotal wrong", "grand total wrong")}
examples, bills_with_error = [], 0
for f in sorted(glob.glob(a.folder + "/*.jpg"))[: a.n]:
    g = json.load(open(f[:-4] + ".json")); im = Image.open(f).convert("RGB")
    with torch.inference_mode(): pred, fields = generate_json(model, processor, im, "cuda", 512, with_conf=True)
    p = ensure_schema(pred); issues = []; low = {fl["path"] for fl in fields if fl["p_min"] < THR}
    for i in range(min(len(p["items"]), len(g["items"]))):
        x, y = p["items"][i], g["items"][i]
        for fld, cat in (("name", "item name misread"), ("qty", "quantity misread"), ("unit_price", "unit price misread"), ("total", "line total misread")):
            bad = nm(x[fld]) != nm(y[fld]) if fld == "name" else num(x[fld]) != num(y[fld])
            if bad: cats[cat] += 1; issues.append({"text": f"{cat}: read {x[fld]!r}, label {y[fld]!r} ({y['name']})", "kind": cat, "item": y["name"], "read": str(x[fld]), "label": str(y[fld]), "path": f"items[{i}].{fld}", "flagged": f"items[{i}].{fld}" in low})
    d = len(p["items"]) - len(g["items"])
    if d < 0: cats["item missed"] += -d; issues.append({"text": f"{-d} item(s) missed", "kind": "item missed", "path": "items", "flagged": False})
    if d > 0: cats["item added"] += d; issues.append({"text": f"{d} item(s) added", "kind": "item added", "path": "items", "flagged": False})
    pc, gc = sorted((c["type"], num(c["amount"])) for c in p["charges"]), sorted((c["type"], num(c["amount"])) for c in g["charges"])
    pt, gt_ = sorted(t for t, _ in pc), sorted(t for t, _ in gc)
    if len(pc) < len(gc): cats["charge missed"] += len(gc) - len(pc); issues.append({"text": "charge line(s) missed", "kind": "charge missed", "path": "charges", "flagged": False})
    elif len(pc) > len(gc): cats["charge added"] += len(pc) - len(gc); issues.append({"text": "charge line(s) added", "kind": "charge added", "path": "charges", "flagged": False})
    elif pt != gt_: cats["charge type wrong"] += 1; issues.append({"text": f"charge types read {pt}, label {gt_}", "kind": "charge type wrong", "read": str(pt), "label": str(gt_), "path": "charges", "flagged": any(q.startswith("charges") for q in low)})
    elif pc != gc: cats["charge amount wrong"] += 1; issues.append({"text": "charge amounts differ", "kind": "charge amount wrong", "path": "charges", "flagged": any(q.startswith("charges") for q in low)})
    if num(p["subtotal"]) != num(g["subtotal"]): cats["subtotal wrong"] += 1; issues.append({"text": "subtotal wrong", "kind": "subtotal wrong", "path": "subtotal", "flagged": "subtotal" in low})
    if num(p["grand_total"]) != num(g["grand_total"]): cats["grand total wrong"] += 1; issues.append({"text": "grand total wrong", "kind": "grand total wrong", "path": "grand_total", "flagged": "grand_total" in low})
    if issues:
        bills_with_error += 1
        examples.append({"bill": os.path.basename(f), "issues": issues})
total = sum(cats.values())
flagged = sum(1 for e in examples for i in e["issues"] if i.get("flagged")); n_issues = sum(len(e["issues"]) for e in examples)
out = {"bills": a.n, "bills_with_any_error": bills_with_error, "error_counts": cats, "errors_total": total, "errors_highlighted_by_the_app": flagged, "errors_listed": n_issues, "examples": examples}
json.dump(out, open(a.out, "w"), indent=1); print("ERR", json.dumps({k: v for k, v in cats.items() if v}), "bills with an error:", bills_with_error)
