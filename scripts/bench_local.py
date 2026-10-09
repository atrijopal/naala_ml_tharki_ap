"""Local inference benchmark of the zero-risk optimisations. Every variant is compared with the CURRENT behaviour: same outputs required.
  python scripts/bench_local.py --ckpt ckpt/b2/.../best [--ckpt_fp16 ckpt/best_fp16] --manifest data/local_test/manifest.jsonl --out bench_local.json"""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from splitsnap.infer import generate_json
from splitsnap.model import load_model
from splitsnap.preprocess import prepare_image


def sync():
    torch.cuda.synchronize()


def ms(xs):
    xs = sorted(xs); return {"p50": round(xs[len(xs) // 2] * 1000, 1), "p95": round(xs[min(int(len(xs) * 0.95), len(xs) - 1)] * 1000, 1), "mean": round(statistics.mean(xs) * 1000, 1)}


def run(model, processor, recs, reps=3, **kw):
    outs, per = [], [[] for _ in recs]
    for rep in range(reps):
        for i, r in enumerate(recs):
            sync(); t = time.perf_counter()
            o = generate_json(model, processor, r["image"], "cuda", 512, task=r["task"], **kw)
            sync(); per[i].append(time.perf_counter() - t)
            if rep == 0: outs.append(json.dumps(o, sort_keys=True, ensure_ascii=False))
    return outs, [statistics.median(p) for p in per]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--ckpt_fp16", default=""); ap.add_argument("--manifest", default="data/local_test/manifest.jsonl")
    ap.add_argument("--out", default="results/bench_local.json"); a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.manifest)]
    phone = [r for r in recs if r["id"].startswith("phone_")]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_images": len(recs), "n_phone_size": len(phone)}
    # ---- start-up: load, then the first request vs later ones (what a warm-up removes)
    t = time.perf_counter(); model, processor = load_model(a.ckpt); model.to("cuda").half().eval(); sync(); res["load_to_gpu_s"] = round(time.perf_counter() - t, 2)
    cold = []
    for k in range(4):
        sync(); t = time.perf_counter(); generate_json(model, processor, recs[0]["image"], "cuda", 512, task=recs[0]["task"]); sync(); cold.append(time.perf_counter() - t)
    res["first_request_ms"] = round(cold[0] * 1000, 1); res["later_requests_ms"] = [round(x * 1000, 1) for x in cold[1:]]
    # ---- photo preparation alone (CPU), 12 MP photos
    if phone:
        old = []; new = []
        for _ in range(5):
            for r in phone:
                t = time.perf_counter(); prepare_image(r["image"]); old.append(time.perf_counter() - t)
                t = time.perf_counter(); prepare_image(r["image"], fast=True); new.append(time.perf_counter() - t)
        res["prep_12mp_current_ms"], res["prep_12mp_fast_ms"] = ms(old), ms(new)
        d = []
        import numpy as np
        for r in phone:
            a1 = np.array(processor(prepare_image(r["image"]), return_tensors="pt").pixel_values[0]); a2 = np.array(processor(prepare_image(r["image"], fast=True), return_tensors="pt").pixel_values[0])
            d.append(float(np.abs(a1 - a2).mean()))
        res["model_input_mean_abs_diff_normalised_units"] = round(statistics.mean(d), 4)
    # ---- variants vs the current behaviour (fp16, current prep, unk blocked)
    base_out, base_t = run(model, processor, recs)
    res["baseline_fp16"] = {"all": ms(base_t), "phone_size": ms([t for r, t in zip(recs, base_t) if r["id"].startswith("phone_")]) if phone else None}
    for name, kw in (("fast_prep", {"fast_prep": True}), ("no_unk_filter", {"block_unk": False}), ("fast_prep+no_unk_filter", {"fast_prep": True, "block_unk": False})):
        o, tt = run(model, processor, recs, **kw)
        same = sum(x == y for x, y in zip(o, base_out))
        res[name] = {"all": ms(tt), "phone_size": ms([t for r, t in zip(recs, tt) if r["id"].startswith("phone_")]) if phone else None, "identical_outputs": f"{same}/{len(recs)}"}
    # ---- fp32 vs fp16
    torch.cuda.reset_peak_memory_stats(); model.float(); o32, t32 = run(model, processor, recs, reps=2)
    res["fp32"] = {"all": ms(t32), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "identical_to_fp16": f"{sum(x == y for x, y in zip(o32, base_out))}/{len(recs)}"}
    model.half(); torch.cuda.reset_peak_memory_stats(); run(model, processor, recs, reps=1); res["fp16_peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    # ---- half-precision checkpoint: size, load time, same answers
    if a.ckpt_fp16:
        size = lambda p: round(sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / 1e6)
        del model; torch.cuda.empty_cache(); t = time.perf_counter(); m2, p2 = load_model(a.ckpt_fp16); m2.to("cuda").half().eval(); sync()
        o16, t16 = run(m2, p2, recs)
        res["fp16_checkpoint"] = {"size_mb": size(a.ckpt_fp16), "fp32_checkpoint_size_mb": size(a.ckpt), "load_to_gpu_s": round(time.perf_counter() - t - sum(t16) * 3, 2),
                                  "identical_outputs": f"{sum(x == y for x, y in zip(o16, base_out))}/{len(recs)}", "all": ms(t16)}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
