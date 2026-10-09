"""Before/after fine-tuning on the same bills: untrained donut-base (+ our schema tokens, same input size) vs our fine-tuned checkpoint.
Bills: the first N of examples/final_run (generated with seeds nobody used before, labels exact). Local GPU, fp16, greedy, max 512 tokens.
  python scripts/before_after.py --ckpt ckpt/best_fp16 --n 30 --out results/before_after.json"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from splitsnap.infer import generate_json
from splitsnap.metrics import evaluate_records
from splitsnap.model import build_model, load_model
from splitsnap.schema import ensure_schema

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=30)
ap.add_argument("--folder", default="examples/final_run"); ap.add_argument("--out", default="results/before_after.json"); a = ap.parse_args()
files = sorted(glob.glob(a.folder + "/*.jpg"))[: a.n]
gts = [json.load(open(f[:-4] + ".json")) for f in files]
res = {"n": len(files), "bills": [os.path.basename(f) for f in files], "gpu": torch.cuda.get_device_name(0)}
preds = {}
for name in ("before", "after"):
    model, processor = build_model(1280, 960, 512) if name == "before" else load_model(a.ckpt)
    model.to("cuda").half().eval(); out, secs = [], []
    for f in files:
        t = time.perf_counter()
        with torch.inference_mode():
            o = generate_json(model, processor, Image.open(f), "cuda", 512)
        secs.append(time.perf_counter() - t); out.append(ensure_schema(o))
    preds[name] = out
    res[name] = {k: round(float(v), 4) for k, v in evaluate_records(out, gts).items() if isinstance(v, (int, float))}
    res[name]["median_s_per_bill"] = round(sorted(secs)[len(secs) // 2], 3)
    print("BA", name, json.dumps(res[name]), flush=True)
    del model; torch.cuda.empty_cache()
res["examples"] = {"gt": gts[:3], "before": preds["before"][:3], "after": preds["after"][:3]}
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
