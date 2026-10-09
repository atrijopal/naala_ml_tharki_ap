"""R2 study: how do accuracy, the retake prompt and the confidence flags behave when photos get worse?
The 100 unseen generated bills (examples/final_run) are degraded in controlled steps (blur, tilt, fading) and read by the model.
Per condition: field/item/charge F1, grand total correct, share of bills the app would ask to retake, and how well the lowest-confidence fields
(p_min below the calibrated threshold) point at the real errors. Generated bills only; not a real-photo result.
  python scripts/degradation_study.py --ckpt ckpt/best_fp16 --out results/degradation.json"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image, ImageFilter
from splitsnap.confidence import auroc, label_fields
from splitsnap.infer import generate_json
from splitsnap.metrics import evaluate_records
from splitsnap.model import load_model
from splitsnap.preprocess import quality_report
from splitsnap.schema import ensure_schema

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--folder", default="examples/final_run")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", default="results/degradation.json"); ap.add_argument("--calibration", default="results/calibration.json"); a = ap.parse_args()
THR = json.load(open(a.calibration))["aggregators"]["p_min"]["flag_below"]


def blur(r):
    return lambda im: im.filter(ImageFilter.GaussianBlur(r))
def tilt(deg):
    return lambda im: im.rotate(deg, expand=True, resample=Image.BICUBIC, fillcolor=(228, 228, 228))
def fade(k):                                              # k = share of the print contrast that is left (1 = unchanged)
    return lambda im: Image.fromarray((255 - (255 - np.asarray(im.convert("RGB")).astype(np.float32)) * k).clip(0, 255).astype(np.uint8))
CONDS = [("clean", None, lambda im: im)] + [("blur", r, blur(r)) for r in (1.5, 3, 5)] + [("tilt", d, tilt(d)) for d in (4, 8, 15)] + [("fade", k, fade(k)) for k in (0.6, 0.4, 0.25)]

files = sorted(glob.glob(a.folder + "/*.jpg"))[: a.n]; gts = [json.load(open(f[:-4] + ".json")) for f in files]
model, processor = load_model(a.ckpt); model.to("cuda").half().eval()
res = {"n_bills": len(files), "flag_threshold_p_min": THR, "gpu": torch.cuda.get_device_name(0), "conditions": []}
for kind, level, fn in CONDS:
    preds, corr, p_min, retake, t0 = [], [], [], 0, time.time()
    for f, g in zip(files, gts):
        im = fn(Image.open(f).convert("RGB")); retake += bool(quality_report(im)["retake"])
        with torch.inference_mode(): pred, fields = generate_json(model, processor, im, "cuda", 512, with_conf=True)
        preds.append(ensure_schema(pred)); ok = label_fields(fields, g); corr.append([int(x) for x in ok]); p_min.append([f_["p_min"] for f_ in fields])
    m = evaluate_records(preds, gts)
    c = np.array([x for r in corr for x in r]); s = np.array([x for r in p_min for x in r]); flagged = s < THR; err = c == 0
    per_bill_err = [sum(1 for x in r if x == 0) for r in corr]
    per_bill_missed = [sum(1 for x, s_ in zip(r, ps) if x == 0 and s_ >= THR) for r, ps in zip(corr, p_min)]
    row = {"kind": kind, "level": level, "field_f1": round(m["field_f1"], 4), "item_f1": round(m["item_f1"], 4), "charge_f1": round(m["charge_f1"], 4), "grand_total_acc": round(m["grand_total_acc"], 4),
           "retake_share": round(retake / len(files), 3), "fields": int(len(c)), "wrong_fields": int(err.sum()), "flagged_share": round(float(flagged.mean()), 4),
           "errors_caught": round(float((flagged & err).sum() / max(err.sum(), 1)), 4) if err.sum() else None,
           "unflagged_correct": round(float(c[~flagged].mean()), 4) if (~flagged).any() else None,
           "auroc": round(float(auroc(s, c.astype(int))), 4) if 0 < err.sum() < len(c) else None,
           "bills_with_an_error": round(sum(1 for e in per_bill_err if e) / len(files), 3), "bills_with_an_unflagged_error": round(sum(1 for e in per_bill_missed if e) / len(files), 3),
           "flags_per_bill": round(float(flagged.sum() / len(files)), 2), "seconds": round(time.time() - t0)}
    res["conditions"].append(row); print("DEG", json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
