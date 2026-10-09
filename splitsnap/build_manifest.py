# splitsnap/build_manifest.py
"""build_manifest.py - merge manifests, remove cross-split duplicate images (leakage), print a summary.
python -m splitsnap.build_manifest data/cord/manifest.jsonl data/synth/manifest_*.jsonl data/real/manifest.jsonl
An image that appears in more than one split (identical bytes, MD5) is kept ONLY in the most protected split
(test > val > train): the lower-split copies are dropped and reported. Identical images inside one split are kept."""
import collections, hashlib, json, sys

PROTECT = {"test": 0, "val": 1, "train": 2}


def dedupe(recs):
    by_hash = collections.defaultdict(list)
    for r in recs:
        by_hash[hashlib.md5(open(r["image"], "rb").read()).hexdigest()].append(r)
    drop = []
    for group in by_hash.values():
        splits = {r["split"] for r in group}
        if len(splits) > 1:
            keep = min(splits, key=lambda s: PROTECT.get(s, 9))
            drop += [r for r in group if r["split"] != keep]
    ids = {r["id"] for r in drop}
    return [r for r in recs if r["id"] not in ids], drop


def main(paths, out="data/manifest.jsonl"):
    recs = [json.loads(l) for p in paths for l in open(p)]
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    recs, dropped = dedupe(recs)
    if dropped:
        c = collections.Counter((r["source"], r["split"]) for r in dropped)
        print(f"LEAKAGE FIX: dropped {len(dropped)} images that also appear in a more protected split: "
              + ", ".join(f"{s}/{sp}={n}" for (s, sp), n in sorted(c.items())))
        for r in dropped[:10]:
            print("   dropped", r["id"])
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["source"], r["split"]) for r in recs)
    print(f"{len(recs)} records -> {out}")
    for (s, sp), n in sorted(c.items()):
        print(f"  {s:6s} {sp:5s} {n}")
    return dropped


if __name__ == "__main__":
    main(sys.argv[1:])
