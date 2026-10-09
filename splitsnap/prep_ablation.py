# splitsnap/prep_ablation.py
"""prep_ablation.py - does an image filter help? Same model (public CORD Donut), same bills, one filter at a time.
python -m splitsnap.prep_ablation --n_cord 20 --n_synth 20 --out prep_ablation.json"""
import argparse, json
from .baseline import pick, run_public_cord

MODES = [None, "clahe", "shadow", "denoise", "sharpen", "deskew", "shadow+clahe", "deskew+clahe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--n_cord", type=int, default=20); ap.add_argument("--n_synth", type=int, default=20)
    ap.add_argument("--out", default="prep_ablation.json")
    a = ap.parse_args()
    allr = [json.loads(l) for l in open(a.manifest)]
    recs = pick(allr, "cord", a.n_cord) + pick(allr, "synth", a.n_synth)
    res = {}
    for m in MODES:
        res[m or "none"], _ = run_public_cord(recs, m)
        print("PREP " + (m or "none") + " " + json.dumps(res[m or "none"]["metrics"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
