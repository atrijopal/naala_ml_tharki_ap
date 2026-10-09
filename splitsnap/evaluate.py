# splitsnap/evaluate.py
"""evaluate.py - run a checkpoint over a split and report metrics per source.
python -m splitsnap.evaluate --ckpt run1/best --manifest data/manifest.jsonl --split test --out eval_test.json"""
import argparse, json, time
import torch
from .model import load_model
from .infer import generate_json
from .metrics import evaluate_records
from .confidence import label_fields
from .metrics_sroie import evaluate_sroie
from .schema import TASK_SROIE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--split", default="test"); ap.add_argument("--sources", default="")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--out", default="eval.json")
    ap.add_argument("--conf", action="store_true", help="also store per-field confidences (for calibration)")
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--use_crop", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(a.ckpt); model.to(dev).eval()
    if dev == "cuda":
        model.half()
    srcs = [s for s in a.sources.split(",") if s]
    recs = [r for r in map(json.loads, open(a.manifest)) if r["split"] == a.split and (not srcs or r["source"] in srcs)]
    recs = recs[:a.n] if a.n else recs
    by, dump, lat = {}, [], []
    for r in recs:
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.time()
        if r["task"] == TASK_SROIE:                                  # auxiliary task: no confidences, own scorer
            pred = generate_json(model, processor, r["image"], dev, a.max_length, use_crop=a.use_crop, task=TASK_SROIE)
            by.setdefault(r["source"], ([], [])); by[r["source"]][0].append(pred); by[r["source"]][1].append(r["gt"])
            dump.append({"id": r["id"], "source": r["source"], "pred": pred, "gt": r["gt"]})
            lat.append(time.time() - t0)
            continue
        out = generate_json(model, processor, r["image"], dev, a.max_length, with_conf=a.conf, use_crop=a.use_crop)
        pred, fields = out if a.conf else (out, None)
        by.setdefault(r["source"], ([], [])); by[r["source"]][0].append(pred); by[r["source"]][1].append(r["gt"])
        item = {"id": r["id"], "source": r["source"], "pred": pred, "gt": r["gt"]}
        if a.conf:
            item["fields"] = fields; item["correct"] = label_fields(fields, r["gt"])
        dump.append(item)
        torch.cuda.synchronize() if dev == "cuda" else None
        lat.append(time.time() - t0)
    res = {s: (evaluate_sroie(p, g) if s == "sroie" else evaluate_records(p, g)) for s, (p, g) in by.items()}
    lat.sort()
    latency = {"n": len(lat), "p50": round(lat[len(lat) // 2], 3), "p95": round(lat[int(len(lat) * 0.95)], 3), "max": round(lat[-1], 3),
               "device": dev, "fp16": dev == "cuda"} if lat else {}
    json.dump({"metrics": res, "latency_sec_per_bill": latency, "predictions": dump}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1)); print("LATENCY", json.dumps(latency))


if __name__ == "__main__":
    main()
