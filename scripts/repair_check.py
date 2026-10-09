"""How much would the one-tap rate suggestion (line total / qty) fix on the 100 unseen generated bills?
Counts, on the same model outputs: field F1 and exact bills before and after applying every suggestion, and whether any correct field was broken.
  python scripts/repair_check.py --ckpt ckpt/best_fp16 --out results/repair_check.json"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from splitsnap.infer import generate_json
from splitsnap.metrics import evaluate_records
from splitsnap.model import load_model
from splitsnap.schema import ensure_schema
from splitsnap.validate import repair_unit_prices

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--folder", default="examples/final_run"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", default="results/repair_check.json"); a = ap.parse_args()
num = lambda x: (lambda s: round(float(s), 2) if s.replace(".", "", 1).replace("-", "", 1).isdigit() else None)(str(x).replace(",", "").strip())
model, processor = load_model(a.ckpt); model.to("cuda").half().eval()
files = sorted(glob.glob(a.folder + "/*.jpg"))[: a.n]; gts = [json.load(open(f[:-4] + ".json")) for f in files]; before, after = [], []
fixed = broken = suggested = 0
for f, g in zip(files, gts):
    with torch.inference_mode(): p = ensure_schema(generate_json(model, processor, Image.open(f).convert("RGB"), "cuda", 512))
    r = repair_unit_prices(p); before.append(p); after.append(r)
    for x, y, gi in zip(p["items"], r["items"], g["items"]):
        if num(x["unit_price"]) != num(y["unit_price"]):
            suggested += 1; fixed += int(num(y["unit_price"]) == num(gi["unit_price"])); broken += int(num(x["unit_price"]) == num(gi["unit_price"]) and num(y["unit_price"]) != num(gi["unit_price"]))
exact = lambda ps: sum(1 for p, g in zip(ps, gts) if [(num(i["unit_price"])) for i in p["items"]] == [num(i["unit_price"]) for i in g["items"]])
mb, ma = evaluate_records(before, gts), evaluate_records(after, gts)
out = {"bills": len(files), "suggestions_made": suggested, "suggestions_that_fix_a_wrong_rate": fixed, "correct_rates_broken": broken,
       "field_f1_before": round(mb["field_f1"], 4), "field_f1_after": round(ma["field_f1"], 4), "bills_with_all_rates_right_before": exact(before), "bills_with_all_rates_right_after": exact(after)}
json.dump(out, open(a.out, "w"), indent=1); print("REPAIR", json.dumps(out))
