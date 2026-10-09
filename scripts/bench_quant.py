"""int8 weight-only quantisation of the decoder (and optionally the encoder) with torchao, local GPU. A variant is accepted only if its
outputs equal eager fp16 on every bill; speed and memory are reported either way.
  python scripts/bench_quant.py --ckpt ckpt/best_fp16 --manifest data/local_test/manifest.jsonl --out results/bench_quant.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_compile import timed, sync
from splitsnap.infer import generate_json
from splitsnap.model import load_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--manifest", default="data/local_test/manifest.jsonl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default="results/bench_quant.json"); ap.add_argument("--variants", default="eager,dec_int8,all_int8")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.manifest)][: a.n]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n": len(recs)}
    ref = None
    for v in a.variants.split(","):
        model, processor = load_model(a.ckpt); model.to("cuda").half().eval()
        try:
            if v != "eager":
                from torchao.quantization import quantize_
                import torchao.quantization as tq
                cfg = {"int8": lambda: tq.Int8WeightOnlyConfig(), "fp8w": lambda: tq.Float8WeightOnlyConfig(), "fp8dyn": lambda: tq.Float8DynamicActivationFloat8WeightConfig()}
                kind = "int8" if "int8" in v else ("fp8dyn" if "fp8dyn" in v else "fp8w")
                quantize_(model.decoder, cfg[kind]())
                if v.startswith("all_"): quantize_(model.encoder, cfg[kind]())
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generate_json(model, processor, recs[0]["image"], "cuda", 512, task=recs[0]["task"]); sync()
                outs, st = timed(model, processor, recs)
            if ref is None: ref = outs
            st.update(peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), identical=f"{sum(x == y for x, y in zip(ref, outs))}/{len(outs)}")
            res[v] = st
        except Exception as e:
            res[v] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print("QUANT", v, json.dumps(res[v]), flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
