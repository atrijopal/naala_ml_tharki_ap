# splitsnap/bench.py
"""bench.py - measure real training speed / memory / inference latency on THIS GPU and forecast total hours.
python -m splitsnap.bench --manifest data/manifest.jsonl --height 1280 --width 960 --steps 24 --grad_ckpt
Prints one 'BENCH {json}' line per configuration (grep for it in the Kaggle log)."""
import argparse, json, time
import torch
from torch.utils.data import DataLoader
from .model import build_model, sanity_check
from .dataset import DonutJSONLDataset
from .infer import generate_json


def run(a):
    dev = "cuda"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, processor = build_model(a.height, a.width, a.max_length)
    sanity_check(model, processor, a.height, a.width)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(); model.decoder.config.use_cache = False
    model.to(dev).train()
    ds = DonutJSONLDataset(a.manifest, "train", processor, a.max_length, augment_on=True)
    loader = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    if a.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=3e-5)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda")
    times, it = [], iter(loader)
    for step in range(a.steps + a.warmup):
        batch = next(it)
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=batch["pixel_values"].to(dev), decoder_input_ids=batch["decoder_input_ids"].to(dev),
                        labels=batch["labels"].to(dev))
        scaler.scale(out.loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= a.warmup:
            times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    sec_per_sample = sum(times) / len(times) / a.bs
    # data-loading check: time to produce one batch with no GPU work
    t = time.time(); [next(it) for _ in range(4)]; load_s = (time.time() - t) / 4 / a.bs
    # inference latency, fp16, greedy
    model.eval().half(); model.decoder.config.use_cache = True
    val = [r for r in map(json.loads, open(a.manifest)) if r["split"] == "val" and r["task"] == "<s_splitsnap>"][:a.val_n]
    lat = []
    for r in val:
        torch.cuda.synchronize(); t = time.time()
        generate_json(model, processor, r["image"], dev, a.max_length)
        torch.cuda.synchronize(); lat.append(time.time() - t)
    lat = lat[1:] or lat
    res = {"height": a.height, "width": a.width, "grad_ckpt": a.grad_ckpt, "adam8bit": a.adam8bit,
           "sec_per_sample": round(sec_per_sample, 3), "peak_train_gb": round(peak, 2),
           "dataload_sec_per_sample": round(load_s, 3),
           "infer_sec_per_bill_p50": round(sorted(lat)[len(lat) // 2], 2) if lat else None,
           "infer_sec_per_bill_max": round(max(lat), 2) if lat else None,
           "hours_per_10k_sample_passes": round(10000 * sec_per_sample / 3600, 2)}
    print("BENCH " + json.dumps(res), flush=True)
    del model, opt; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--height", type=int, default=1280); ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--bs", type=int, default=1); ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--val_n", type=int, default=6)
    ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--adam8bit", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
