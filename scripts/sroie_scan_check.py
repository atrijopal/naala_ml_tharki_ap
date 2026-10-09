"""Does the main (items and charges) task read real SROIE scans, and do retries with image enhancement help?
The 8 local SROIE scans have only four labels (company, date, address, total), so for the main task we can check whether items came out at all and
whether the grand total equals the SROIE total. The auxiliary <s_sroie> task is run for reference.   python scripts/sroie_scan_check.py --ckpt ckpt/best_fp16"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from splitsnap.infer import generate_json
from splitsnap.model import load_model
from splitsnap.preprocess import enhance_image
from splitsnap.schema import TASK_SROIE, ensure_schema

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--folder", default="examples/sroie"); ap.add_argument("--out", default="results/sroie_scan_check.json"); a = ap.parse_args()
VARIANTS = ["none", "clahe", "shadow", "sharpen", "denoise", "deskew", "shadow+clahe"]
num = lambda x: (lambda s: float(s) if s.replace(".", "", 1).replace("-", "", 1).isdigit() else None)(str(x).replace(",", "").strip())
norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
model, processor = load_model(a.ckpt); model.to("cuda").half().eval()
rows = []
for f in sorted(glob.glob(a.folder + "/*.jpg")):
    gt = json.load(open(f[:-4] + ".json")); img = Image.open(f).convert("RGB"); r = {"scan": os.path.basename(f), "label_total": gt.get("total"), "main_task": {}}
    for v in VARIANTS:
        im = img if v == "none" else enhance_image(img.copy(), v)
        with torch.inference_mode(): p = ensure_schema(generate_json(model, processor, im, "cuda", 512))
        r["main_task"][v] = {"items": len(p["items"]), "charges": len(p["charges"]), "grand_total": p["grand_total"], "total_matches_label": bool(num(p["grand_total"]) is not None and num(p["grand_total"]) == num(gt.get("total")))}
    with torch.inference_mode(): aux = generate_json(model, processor, img, "cuda", 512, task=TASK_SROIE)
    r["aux_task"] = {k: {"read": aux.get(k), "exact": norm(aux.get(k, "")) == norm(gt.get(k, "")) if k != "total" else num(aux.get(k)) == num(gt.get("total"))} for k in ("company", "date", "address", "total")}
    rows.append(r); print("SROIE", r["scan"], {v: (x["items"], x["total_matches_label"]) for v, x in r["main_task"].items()}, flush=True)
n = len(rows)
summ = {"scans": n, "main_task_nonempty": {v: sum(1 for r in rows if r["main_task"][v]["items"] > 0) for v in VARIANTS},
        "main_task_total_matches": {v: sum(1 for r in rows if r["main_task"][v]["total_matches_label"]) for v in VARIANTS},
        "first_nonempty_in_order": sum(1 for r in rows if any(r["main_task"][v]["items"] > 0 for v in VARIANTS)),
        "aux_task_fields_exact": {k: sum(1 for r in rows if r["aux_task"][k]["exact"]) for k in ("company", "date", "address", "total")}}
json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1, ensure_ascii=False); print("SROIE-SUMMARY", json.dumps(summ))
