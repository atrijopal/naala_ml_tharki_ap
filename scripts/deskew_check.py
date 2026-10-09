"""Does straightening tilted photos before reading help? Same 100 unseen generated bills, tilted by 0/4/8/15 degrees, read with and without the deskew filter.
  python scripts/deskew_check.py --ckpt ckpt/best_fp16 --out results/deskew_check.json"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from splitsnap.infer import generate_json
from splitsnap.metrics import evaluate_records
from splitsnap.model import load_model
from splitsnap.preprocess import enhance_image, estimate_skew
from splitsnap.schema import ensure_schema

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--folder", default="examples/final_run"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", default="results/deskew_check.json"); a = ap.parse_args()
files = sorted(glob.glob(a.folder + "/*.jpg"))[: a.n]; gts = [json.load(open(f[:-4] + ".json")) for f in files]
model, processor = load_model(a.ckpt); model.to("cuda").half().eval(); rows = []
for deg in (0, 4, 8, 15):
    ims = [Image.open(f).convert("RGB") for f in files]
    if deg: ims = [im.rotate(deg, expand=True, resample=Image.BICUBIC, fillcolor=(228, 228, 228)) for im in ims]
    skews = [estimate_skew(im) for im in ims]
    for mode in ("as is", "deskew"):
        preds = []
        for im in ims:
            x = im if mode == "as is" else enhance_image(im.copy(), "deskew")
            with torch.inference_mode(): preds.append(ensure_schema(generate_json(model, processor, x, "cuda", 512)))
        m = evaluate_records(preds, gts)
        rows.append({"tilt_deg": deg, "mode": mode, "median_estimated_skew": round(sorted(skews)[len(skews) // 2], 1), "field_f1": round(m["field_f1"], 4), "item_f1": round(m["item_f1"], 4), "charge_f1": round(m["charge_f1"], 4), "grand_total_acc": round(m["grand_total_acc"], 4)})
        print("DESKEW", json.dumps(rows[-1]), flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
