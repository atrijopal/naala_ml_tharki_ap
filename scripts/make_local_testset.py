"""Build data/local_test/: a small labelled test set for local inference benchmarks, from examples/ (real SROIE scans, artificial bills)
plus 'phone-size' 12-megapixel JPEG photos of the real scans (a scan placed on a table texture), to test the photo-preparation path.
python scripts/make_local_testset.py"""
import glob, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from splitsnap.augment import _table_texture
from splitsnap.schema import TASK_SROIE, TASK_START

OUT = "data/local_test"; os.makedirs(f"{OUT}/phone", exist_ok=True)
recs = []
for p in sorted(glob.glob("examples/sroie/*.jpg")):
    gt = json.load(open(p.replace(".jpg", ".json")))
    recs.append({"id": os.path.basename(p), "image": p, "gt": gt, "task": TASK_SROIE, "source": "sroie", "split": "test"})
for p in sorted(glob.glob("examples/synthetic_augmented/*.jpg")):
    i = os.path.basename(p).split("_")[1]
    gt = json.load(open(f"examples/synthetic_flat/synth_{i}.json"))
    recs.append({"id": os.path.basename(p), "image": p, "gt": gt, "task": TASK_START, "source": "synth", "split": "test"})
rng = random.Random(5)
for r in [x for x in recs if x["source"] == "sroie"][:4]:               # phone-size photos of real scans
    scan = Image.open(r["image"]).convert("RGB"); h = 3700; w = int(scan.width * h / scan.height)
    canvas = _table_texture(3060, 4080, rng); canvas.paste(scan.resize((w, h), Image.BICUBIC), ((3060 - w) // 2, 190))
    out = f"{OUT}/phone/phone_{r['id']}"; canvas.save(out, quality=90)
    recs.append({"id": "phone_" + r["id"], "image": out, "gt": r["gt"], "task": TASK_SROIE, "source": "sroie", "split": "test"})
with open(f"{OUT}/manifest.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(recs), "records ->", f"{OUT}/manifest.jsonl", "| phone-size photos:", sum(r["id"].startswith("phone_") for r in recs))
