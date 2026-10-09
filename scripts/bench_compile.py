"""torch.compile experiments for inference (local GPU). Every variant must give the SAME outputs as eager fp16, else it is rejected.
  python scripts/bench_compile.py --ckpt ckpt/best_fp16 --manifest data/local_test/manifest.jsonl --out results/bench_compile.json"""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from splitsnap.infer import generate_json
from splitsnap.model import load_model


def sync():
    torch.cuda.synchronize()


def timed(model, processor, recs, reps=3):
    outs, per = [], [[] for _ in recs]
    for rep in range(reps):
        for i, r in enumerate(recs):
            sync(); t = time.perf_counter()
            o = generate_json(model, processor, r["image"], "cuda", 512, task=r["task"])
            sync(); per[i].append(time.perf_counter() - t)
            if rep == 0: outs.append(json.dumps(o, sort_keys=True, ensure_ascii=False))
    med = [statistics.median(p) for p in per]
    return outs, {"p50_ms": round(1000 * sorted(med)[len(med) // 2], 1), "mean_ms": round(1000 * statistics.mean(med), 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--manifest", default="data/local_test/manifest.jsonl")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default="results/bench_compile.json"); ap.add_argument("--variants", default="eager,enc,enc_dec")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.manifest) if json.loads(l)["id"].startswith("phone_")][: a.n]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n": len(recs)}
    ref = None
    for v in a.variants.split(","):
        model, processor = load_model(a.ckpt); model.to("cuda").half().eval()
        try:
            t0 = time.perf_counter()
            if v in ("enc", "enc_dec"):
                model.encoder = torch.compile(model.encoder)
            if v == "enc_dec":
                model.decoder = torch.compile(model.decoder, dynamic=True)
            with torch.inference_mode():
                generate_json(model, processor, recs[0]["image"], "cuda", 512, task=recs[0]["task"]); sync()   # first call compiles
                compile_s = time.perf_counter() - t0
                outs, st = timed(model, processor, recs)
            if ref is None: ref = outs
            st.update(first_call_incl_compile_s=round(compile_s, 1), identical=f"{sum(x == y for x, y in zip(ref, outs))}/{len(outs)}")
            res[v] = st
        except Exception as e:
            res[v] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print("COMPILE", v, json.dumps(res[v]), flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
