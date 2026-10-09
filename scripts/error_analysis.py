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
    p = ensure_schema(pred); issues = []
    for i in range(min(len(p["items"]), len(g["items"]))):
        x, y = p["items"][i], g["items"][i]
        for fld, cat in (("name", "item name misread"), ("qty", "quantity misread"), ("unit_price", "unit price misread"), ("total", "line total misread")):
            bad = nm(x[fld]) != nm(y[fld]) if fld == "name" else num(x[fld]) != num(y[fld])
            if bad: cats[cat] += 1; issues.append(f"{cat}: read {x[fld]!r}, label {y[fld]!r} ({y['name']})")
    d = len(p["items"]) - len(g["items"])
    if d < 0: cats["item missed"] += -d; issues.append(f"{-d} item(s) missed")
    if d > 0: cats["item added"] += d; issues.append(f"{d} item(s) added")
    pc, gc = sorted((c["type"], num(c["amount"])) for c in p["charges"]), sorted((c["type"], num(c["amount"])) for c in g["charges"])
    pt, gt_ = sorted(t for t, _ in pc), sorted(t for t, _ in gc)
    if len(pc) < len(gc): cats["charge missed"] += len(gc) - len(pc); issues.append("charge line(s) missed")
    elif len(pc) > len(gc): cats["charge added"] += len(pc) - len(gc); issues.append("charge line(s) added")
    elif pt != gt_: cats["charge type wrong"] += 1; issues.append(f"charge types read {pt}, label {gt_}")
    elif pc != gc: cats["charge amount wrong"] += 1; issues.append(f"charge amounts read {[a_ for _, a_ in pc]}, label {[a_ for _, a_ in gc]}")
    if num(p["subtotal"]) != num(g["subtotal"]): cats["subtotal wrong"] += 1; issues.append(f"subtotal read {p['subtotal']}, label {g['subtotal']}")
    if num(p["grand_total"]) != num(g["grand_total"]): cats["grand total wrong"] += 1; issues.append(f"grand total read {p['grand_total']}, label {g['grand_total']}")
    if issues:
        bills_with_error += 1; low = [fl["path"] for fl in fields if fl["p_min"] < THR]
        examples.append({"bill": os.path.basename(f), "issues": issues, "flagged_fields": low})
total = sum(cats.values())
out = {"bills": a.n, "bills_with_any_error": bills_with_error, "error_counts": cats, "errors_total": total, "examples": examples[:12]}
json.dump(out, open(a.out, "w"), indent=1); print("ERR", json.dumps({k: v for k, v in cats.items() if v}), "bills with an error:", bills_with_error)
