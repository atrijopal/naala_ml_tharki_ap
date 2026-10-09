# splitsnap/baseline_vlm.py
"""baseline_vlm.py - zero-shot baseline for ONE open vision-language model on the same bills, prompt and metrics.
python -m splitsnap.baseline_vlm --model Qwen/Qwen2.5-VL-3B-Instruct --n_cord 20 --n_synth 20 --out vlm/qwen.json"""
import argparse, gc, json, math, time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from .baseline import pick
from .metrics import evaluate_records
from .preprocess import prepare_image
from .vlm_utils import PROMPT, extract_json, normalize_pred


def cap_pixels(img, max_pixels):
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(int(w * s), 28), max(int(h * s), 28)), Image.LANCZOS)


def load(model_id, dtype):
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype, device_map="cuda",
                                                        attn_implementation="sdpa").eval()
    return model, proc


@torch.inference_mode()
def ask(model, proc, img, max_new):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    try:
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt")
    except Exception:                                    # older processors: template text + explicit images
        ph = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(ph, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def run(model_id, dtype, recs, a):
    model, proc = load(model_id, dtype)
    torch.cuda.reset_peak_memory_stats()
    preds, raws, lat, fails, done, t_start = [], [], [], 0, [], time.time()
    for i, r in enumerate(recs):
        if (time.time() - t_start) > a.max_minutes * 60:
            print(f"time budget hit after {i} bills", flush=True); break
        img = cap_pixels(prepare_image(r["image"]), a.max_pixels)
        torch.cuda.synchronize(); t = time.time()
        try:
            raw = ask(model, proc, img, a.max_new)
        except Exception as e:
            raw = f"ERROR {e!r}"[:300]
        torch.cuda.synchronize(); lat.append(time.time() - t)
        obj = extract_json(raw)
        fails += obj is None
        preds.append(normalize_pred(obj)); raws.append(raw); done.append(r)
        if i == 2 and fails == 3 and dtype == torch.float16:
            print("first 3 outputs unparseable in fp16 -> will retry in bf16", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache(); return None
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model; gc.collect(); torch.cuda.empty_cache()
    return preds, raws, lat, fails, done, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--n_cord", type=int, default=20); ap.add_argument("--n_synth", type=int, default=20)
    ap.add_argument("--max_pixels", type=int, default=1_400_000); ap.add_argument("--max_new", type=int, default=700)
    ap.add_argument("--max_minutes", type=float, default=20.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allr = [json.loads(l) for l in open(a.manifest)]
    recs = pick(allr, "cord", a.n_cord) + pick(allr, "synth", a.n_synth)
    res = run(a.model, torch.float16, recs, a); dtype = "fp16"
    if res is None:
        res = run(a.model, torch.bfloat16, recs, a); dtype = "bf16"
    preds, raws, lat, fails, done, peak = res
    by = {}
    for p, r in zip(preds, done):
        by.setdefault(r["source"], ([], [])); by[r["source"]][0].append(p); by[r["source"]][1].append(r["gt"])
    lat_s = sorted(lat)
    out = {"model": a.model, "dtype": dtype, "n_scored": len(done), "parse_failures": fails,
           "metrics": {s: evaluate_records(p, g) for s, (p, g) in by.items()},
           "sec_per_bill_p50": round(lat_s[len(lat_s) // 2], 2) if lat_s else None,
           "sec_per_bill_max": round(lat_s[-1], 2) if lat_s else None, "peak_gb": round(peak, 2),
           "example_raw": raws[:2], "example_pred": preds[:2]}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("VLM " + json.dumps({k: v for k, v in out.items() if not k.startswith("example")}), flush=True)


if __name__ == "__main__":
    main()
