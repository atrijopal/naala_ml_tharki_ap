# splitsnap/convert_sroie.py
"""convert_sroie.py - SROIE -> auxiliary task (company/date/address/total) under its OWN start token <s_sroie>.
Do NOT train SROIE samples with the main <s_splitsnap> token: they have no line items and would teach the model to
leave items out. Layout verified by EDA on both Kaggle copies: SROIE2019/{train,test}/{img,entities,box}; the test split
has labels. SROIE has no val split, so `val_n` bills are carved out of train (fixed seed).
python -m splitsnap.convert_sroie --out data/sroie_manifest.jsonl --val_n 60        (finds SROIE2019 under /kaggle/input)"""
import argparse, glob, json, os, random
from .eda import read_entities
from .schema import parse_money, canon_num, TASK_SROIE

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG")


def find_root(base="/kaggle/input"):
    roots = sorted(glob.glob(f"{base}/**/SROIE2019", recursive=True))
    return roots[0] if roots else None


def _dir(parent, names):
    for n in names:
        p = os.path.join(parent, n)
        if os.path.isdir(p):
            return p


def sroie_gt(e):
    try:
        m = parse_money(e.get("total"))
        total = canon_num(m) if m is not None else ""
    except Exception:
        total = ""
    return {"company": str(e.get("company", "")).strip(), "date": str(e.get("date", "")).strip(),
            "address": str(e.get("address", "")).strip(), "total": total}


def export_sroie(root, out_jsonl, val_n=60, seed=0):
    recs = []
    for sp in sorted(os.listdir(root)):
        d = os.path.join(root, sp)
        imd, end = _dir(d, ["img", "image", "images"]) if os.path.isdir(d) else None, _dir(d, ["entities", "key", "labels"]) if os.path.isdir(d) else None
        if not imd or not end:
            continue
        split = "test" if sp.lower().startswith("test") else "train"
        for f in sorted(os.listdir(imd)):
            stem, ext = os.path.splitext(f)
            ep = os.path.join(end, stem + ".txt")
            if ext not in IMG_EXTS or not os.path.exists(ep):
                continue
            e = read_entities(ep)
            gt = sroie_gt(e) if e else None
            if not gt or not gt["total"]:                       # a record without a readable total is not useful for scoring
                continue
            recs.append({"id": f"sroie_{split}_{stem}", "image": os.path.join(imd, f), "gt": gt, "task": TASK_SROIE,
                         "source": "sroie", "split": split})
    train = [r for r in recs if r["split"] == "train"]
    for r in random.Random(seed).sample(train, min(val_n, len(train))):
        r["split"] = "val"
    with open(out_jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = {s: sum(r["split"] == s for r in recs) for s in ("train", "val", "test")}
    print(f"SROIE: {len(recs)} records {c} from {root} -> {out_jsonl}")
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=""); ap.add_argument("--out", default="data/sroie_manifest.jsonl")
    ap.add_argument("--val_n", type=int, default=60); ap.add_argument("--base", default="/kaggle/input")
    a = ap.parse_args()
    root = a.root or find_root(a.base)
    if not root:
        raise SystemExit(f"SROIE2019 folder not found under {a.base}: attach it with EXTRA_DATASETS=urbikn/sroie-datasetv2")
    export_sroie(root, a.out, a.val_n)
