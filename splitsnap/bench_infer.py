# splitsnap/bench_infer.py
"""bench_infer.py - inference benchmark of a trained checkpoint: fp32 vs fp16, with a per-stage time breakdown.
Per bill: preparation (decode/resize/processor, CPU), host->GPU copy, image encoder alone, whole generate() (encoder + token-by-token
decoder), parsing. decoder time ~ generate - encoder. Also peak GPU memory, tokens generated, and whether fp16 changes the answers.
python -m splitsnap.bench_infer --ckpt runs/run1/best --manifest data/manifest.jsonl --out bench_infer.json"""
import argparse, json, random, statistics, time
import torch
from .infer import parse_sequence
from .metrics import evaluate_records
from .metrics_sroie import evaluate_sroie
from .model import load_model
from .preprocess import prepare_image
from .schema import TASK_SROIE, ensure_schema


def sync():
    torch.cuda.synchronize()


def stats(xs):
    xs = sorted(xs)
    return {"p50": round(xs[len(xs) // 2] * 1000, 1), "p95": round(xs[int(len(xs) * 0.95)] * 1000, 1), "mean": round(statistics.mean(xs) * 1000, 1)}


@torch.inference_mode()
def run(model, processor, recs, dtype, max_length, warm=3):
    tok, dev = processor.tokenizer, "cuda"
    rows, preds = [], []
    torch.cuda.reset_peak_memory_stats()
    for k, r in enumerate(recs):
        t0 = time.perf_counter()
        pv = processor(prepare_image(r["image"]), return_tensors="pt").pixel_values
        t1 = time.perf_counter()
        pv = pv.to(dev, dtype=dtype); sync(); t2 = time.perf_counter()
        model.encoder(pv); sync(); t3 = time.perf_counter()                       # encoder alone
        dec = torch.tensor([[tok.convert_tokens_to_ids(r["task"])]], device=dev)
        out = model.generate(pv, decoder_input_ids=dec, max_length=max_length, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             use_cache=True, num_beams=1, bad_words_ids=[[tok.unk_token_id]]); sync(); t4 = time.perf_counter()
        parsed = parse_sequence(processor, tok.decode(out[0], skip_special_tokens=False))
        pred = ensure_schema(parsed) if r["task"] != TASK_SROIE else parsed
        t5 = time.perf_counter()
        preds.append(pred)
        if k >= warm:                                                              # the first bills pay CUDA start-up costs
            ntok = int(out.shape[1]) - 1
            rows.append({"source": r["source"], "prep": t1 - t0, "h2d": t2 - t1, "enc": t3 - t2, "gen": t4 - t3, "dec": max((t4 - t3) - (t3 - t2), 0),
                         "parse": t5 - t4, "total": (t5 - t0) - (t3 - t2), "ntok": ntok})
    return rows, preds, torch.cuda.max_memory_allocated() / 2**30


def score(preds, recs):
    by = {}
    for p, r in zip(preds, recs):
        by.setdefault(r["source"], ([], [])); by[r["source"]][0].append(p); by[r["source"]][1].append(r["gt"])
    return {s: (evaluate_sroie(p, g) if s == "sroie" else evaluate_records(p, g)) for s, (p, g) in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--n_cord", type=int, default=50); ap.add_argument("--n_sroie", type=int, default=50); ap.add_argument("--n_synth", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=512); ap.add_argument("--out", default="results/bench_infer.json")
    a = ap.parse_args()
    allr = [json.loads(l) for l in open(a.manifest)]
    recs = []
    for src, n in (("cord", a.n_cord), ("sroie", a.n_sroie), ("synth", a.n_synth)):
        pool = [r for r in allr if r["split"] == "test" and r["source"] == src]
        random.Random(0).shuffle(pool); recs += pool[:n]
    model, processor = load_model(a.ckpt); model.to("cuda").eval()
    res, all_preds = {"n_bills": len(recs), "gpu": torch.cuda.get_device_name(0)}, {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        model = model.float() if dtype == torch.float32 else model.half()
        rows, preds, peak = run(model, processor, recs, dtype, a.max_length)
        all_preds[name] = preds
        res[name] = {"peak_gpu_gb": round(peak, 2), "n_timed": len(rows), "tokens_per_bill_p50": sorted(r["ntok"] for r in rows)[len(rows) // 2],
                     **{k: stats([r[k] for r in rows]) for k in ("prep", "enc", "dec", "parse", "total")},
                     "decode_ms_per_token": round(1000 * sum(r["dec"] for r in rows) / max(sum(r["ntok"] for r in rows), 1), 2),
                     "accuracy": score(preds, recs)}
        print("BENCHINFER " + name + " " + json.dumps({k: v for k, v in res[name].items() if k != "accuracy"}), flush=True)
    same = sum(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(all_preds["fp32"], all_preds["fp16"]))
    res["fp16_vs_fp32_identical_outputs"] = f"{same}/{len(recs)}"
    json.dump(res, open(a.out, "w"), indent=1)
    print("BENCHINFER identical outputs fp16 vs fp32:", res["fp16_vs_fp32_identical_outputs"], flush=True)


if __name__ == "__main__":
    main()
