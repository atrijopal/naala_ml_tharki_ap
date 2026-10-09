# splitsnap/baseline.py
"""baseline.py - the 'before fine-tuning' numbers (guide section 12), on a SMALL sample.
  B0  donut-base + our new tokens, untrained        (what 'before' means for R1)
  B1  naver-clova-ix/donut-base-finetuned-cord-v2   (someone else's CORD model; label it as such)
Both are scored with the same metrics against the same ground truth, per source (cord / synth).
python -m splitsnap.baseline --manifest data/manifest.jsonl --n_cord 30 --n_synth 30 --out baseline.json"""
import argparse, json, random, re, time
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from .convert_cord import cord_to_schema
from .infer import generate_json
from .metrics import evaluate_records
from .model import build_model, sanity_check
from .preprocess import prepare_image
from .schema import ensure_schema

PUBLIC_CORD = "naver-clova-ix/donut-base-finetuned-cord-v2"
EMPTY = {"items": [], "charges": [], "subtotal": "", "grand_total": ""}


def pick(recs, source, n, seed=0):
    r = [x for x in recs if x["source"] == source and x["split"] == "test" and x["task"] == "<s_splitsnap>"]
    random.Random(seed).shuffle(r)
    return r[:n]


def score(preds, recs, lat, errors):
    by = {}
    for p, r in zip(preds, recs):
        by.setdefault(r["source"], ([], []))
        by[r["source"]][0].append(p); by[r["source"]][1].append(r["gt"])
    res = {s: evaluate_records(p, g) for s, (p, g) in by.items()}
    lat = sorted(lat)
    return {"metrics": res, "n_parse_errors": errors,
            "sec_per_bill_p50": round(lat[len(lat) // 2], 2) if lat else None,
            "sec_per_bill_max": round(lat[-1], 2) if lat else None}


def run_untrained(recs, max_length):
    model, processor = build_model(1280, 960, 768)
    sanity_check(model, processor, 1280, 960)
    model.half().cuda().eval()
    preds, lat, errors, raw = [], [], 0, []
    for r in recs:
        torch.cuda.synchronize(); t = time.time()
        try:
            preds.append(generate_json(model, processor, r["image"], "cuda", max_length))
        except Exception as e:                                   # garbage output may not parse - that is the point
            preds.append(dict(EMPTY)); errors += 1; raw.append(repr(e)[:200])
        torch.cuda.synchronize(); lat.append(time.time() - t)
    out = score(preds, recs, lat, errors); out["example_errors"] = raw[:3]
    del model; torch.cuda.empty_cache()
    return out, preds


@torch.inference_mode()
def run_public_cord(recs, enhance=None):
    proc = DonutProcessor.from_pretrained(PUBLIC_CORD)
    model = VisionEncoderDecoderModel.from_pretrained(PUBLIC_CORD).half().cuda().eval()
    tok = proc.tokenizer
    prompt = tok("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    preds, lat, errors, raw = [], [], 0, []
    for r in recs:
        torch.cuda.synchronize(); t = time.time()
        try:
            pv = proc(prepare_image(r["image"], False, enhance), return_tensors="pt").pixel_values.half().cuda()
            out = model.generate(pv, decoder_input_ids=prompt, max_length=model.decoder.config.max_position_embeddings,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True,
                                 bad_words_ids=[[tok.unk_token_id]], num_beams=1, return_dict_in_generate=True)
            seq = tok.decode(out.sequences[0]).replace(tok.eos_token, "").replace(tok.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            preds.append(ensure_schema(cord_to_schema(proc.token2json(seq)) or EMPTY))
        except Exception as e:
            preds.append(dict(EMPTY)); errors += 1; raw.append(repr(e)[:200])
        torch.cuda.synchronize(); lat.append(time.time() - t)
    out = score(preds, recs, lat, errors); out["example_errors"] = raw[:3]
    del model; torch.cuda.empty_cache()
    return out, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--n_cord", type=int, default=30); ap.add_argument("--n_synth", type=int, default=30)
    ap.add_argument("--n_untrained", type=int, default=12, help="untrained model babbles to max_length: keep this small")
    ap.add_argument("--out", default="baseline.json")
    a = ap.parse_args()
    allr = [json.loads(l) for l in open(a.manifest)]
    recs = pick(allr, "cord", a.n_cord) + pick(allr, "synth", a.n_synth)
    print(f"baseline sample: {len(recs)} test bills ({sum(r['source'] == 'cord' for r in recs)} cord, "
          f"{sum(r['source'] == 'synth' for r in recs)} synth)", flush=True)
    res = {}
    res["B1_public_cord_model"], p1 = run_public_cord(recs)
    print("BASELINE B1 " + json.dumps(res["B1_public_cord_model"]), flush=True)
    small = pick(allr, "cord", a.n_untrained // 2) + pick(allr, "synth", a.n_untrained // 2)
    if small:
        res["B0_untrained_donut_base"], _ = run_untrained(small, 256)
        print("BASELINE B0 " + json.dumps(res["B0_untrained_donut_base"]), flush=True)
    # a few side-by-side examples for eyeballing
    res["examples"] = [{"id": r["id"], "gt": r["gt"], "B1_pred": p} for r, p in list(zip(recs, p1))[:2]
                       + list(zip(recs, p1))[-2:]]
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
