"""Build examples/: a few real and generated images of every kind used in the project, with labels and a guide.
Needs internet (Hugging Face rows API for CORD, `kaggle datasets download -f` for single SROIE files) and the light venv.
  python scripts/make_examples.py [--out examples]
Downloaded files are only opened as images / read as text; nothing from them is executed."""
import argparse, json, os, random, shutil, subprocess, sys, urllib.request, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from splitsnap import synth
from splitsnap.augment import PROFILES, augment
from splitsnap.convert_cord import cord_to_schema
from splitsnap.convert_sroie import sroie_gt
from splitsnap.eda import read_entities
from splitsnap.preprocess import load_image, cap_size

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="examples"); a = ap.parse_args()
OUT = a.out
for d in ("cord", "sroie", "synthetic_flat", "synthetic_augmented", "augmentation_by_profile", "app_screens"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
log = []


def fetch(url, tries=5):
    """urlopen with retries: the Hugging Face rows API answers 500 now and then."""
    import time
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "splitsnap-examples"}), timeout=90).read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def save(img, path, q=88):
    img = cap_size(img.convert("RGB"), 1800)
    img.save(path, quality=q); return img


# ---------------------------------------------------------------- CORD (real receipts, Indonesian) via the rows API
cord_imgs = []
try:
    for split, n in (("train", 6), ("test", 2)):
        url = f"https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix/cord-v2&config=default&split={split}&offset=0&length={n}"
        rows = json.loads(fetch(url))["rows"]
        for r in rows:
            row = r["row"]; i = r["row_idx"]
            raw = fetch(row["image"]["src"])
            p = f"{OUT}/cord/cord_{split}_{i:03d}.jpg"; open(p, "wb").write(raw)
            img = save(Image.open(p), p)
            gt = cord_to_schema(json.loads(row["ground_truth"])["gt_parse"])
            json.dump(gt, open(p.replace(".jpg", ".json"), "w"), indent=1, ensure_ascii=False)
            cord_imgs.append(img); log.append(f"cord/{os.path.basename(p)}  ({split} #{i}, {img.size[0]}x{img.size[1]})")
except Exception as ex:                                   # real images are a bonus: never block the artificial ones
    log.append(f"(real CORD examples skipped: {type(ex).__name__}: {str(ex)[:80]})")

# ---------------------------------------------------------------- SROIE (real scans, Malaysian): single files by name
TMP = f"{OUT}/_tmp"; os.makedirs(TMP, exist_ok=True)
sroie_imgs = []


def kaggle_file(path):
    subprocess.run(["kaggle", "datasets", "download", "-d", "urbikn/sroie-datasetv2", "-f", path, "-p", TMP, "--force"],
                   check=True, capture_output=True, timeout=300)
    base = os.path.join(TMP, os.path.basename(path))
    if not os.path.exists(base) and os.path.exists(base + ".zip"):
        zipfile.ZipFile(base + ".zip").extractall(TMP)
    return base


for split, ids in (("train", ["X51006329399", "X51006332575", "X51006328913", "X51007135247"]),
                   ("test", ["X51005200931", "X51005230605", "X51005230616", "X00016469670"])):
    for sid in ids:
        try:
            imgp = kaggle_file(f"SROIE2019/{split}/img/{sid}.jpg"); entp = kaggle_file(f"SROIE2019/{split}/entities/{sid}.txt")
            dst = f"{OUT}/sroie/sroie_{split}_{sid}.jpg"
            img = save(Image.open(imgp), dst)
            e = read_entities(entp)
            json.dump(sroie_gt(e) if e else {}, open(dst.replace(".jpg", ".json"), "w"), indent=1, ensure_ascii=False)
            sroie_imgs.append(img); log.append(f"sroie/{os.path.basename(dst)}  ({split}, {img.size[0]}x{img.size[1]})")
        except Exception as ex:
            log.append(f"(skipped SROIE {split}/{sid}: {type(ex).__name__})")
shutil.rmtree(TMP, ignore_errors=True)

# ---------------------------------------------------------------- synthetic bills: as generated, then as training sees them
rng = random.Random(2026); flats = []
for i in range(8):
    gt, meta = synth.make_bill(rng)
    flat = synth.render_bill(meta, rng); flats.append((gt, flat))
    p = f"{OUT}/synthetic_flat/synth_{i}.png"; flat.save(p); json.dump(gt, open(p.replace(".png", ".json"), "w"), indent=1)
    aug = augment(flat, random.Random(500 + i), profile="synth")
    save(aug, f"{OUT}/synthetic_augmented/synth_{i}_augmented.jpg")
log += [f"synthetic_flat/synth_0..7.png (+ .json labels): bills straight from the generator",
        "synthetic_augmented/synth_0..7_augmented.jpg: the same bills after the 'synth' training augmentation"]

# ---------------------------------------------------------------- what each augmentation profile does (real image if we have one, else an artificial bill)
demo = flats[0][1]
for name, real in (("cord", cord_imgs[1] if len(cord_imgs) > 1 else None), ("sroie", sroie_imgs[0] if sroie_imgs else None), ("synth", None)):
    src = real if real is not None else demo
    save(src, f"{OUT}/augmentation_by_profile/{name}_original.jpg")
    for k in range(3):
        save(augment(src.copy(), random.Random(40 + k), profile=name), f"{OUT}/augmentation_by_profile/{name}_profile_example_{k}.jpg")
log.append("augmentation_by_profile/: original + 3 random augmentations for each profile (" + ", ".join(PROFILES) + "); uses a real image where one was available, otherwise an artificial bill")

# ---------------------------------------------------------------- app screens (from tests/e2e_ui.py runs in out/ui)
shots = {"light-phone-1-photo": "1_photo", "light-phone-2-check-messy": "2_check_messy", "light-phone-4-assign": "4_assign",
         "light-phone-5-split-override": "5_split_override", "dark-phone-5-split": "5_split_dark", "light-wide-2-check-messy": "wide_2_check"}
for src, dst in shots.items():
    p = f"out/ui/{src}.png"
    if os.path.exists(p):
        shutil.copy(p, f"{OUT}/app_screens/{dst}.png")
log.append("app_screens/: screenshots of the mobile web app (phone light/dark, wide layout)")

open(f"{OUT}/README.md", "w").write("""# Image examples

| Folder | What is in it |
|---|---|
| `cord/` | Real CORD receipts (Indonesian restaurants, phone photos) with our converted label (`.json`) beside each image |
| `sroie/` | Real SROIE scans (Malaysian receipts) with the four SROIE fields (`.json`); note the handwritten totals and name overlays |
| `synthetic_flat/` | Indian-style bills straight from `splitsnap/synth.py`, with exact labels |
| `synthetic_augmented/` | The same bills after the `synth` augmentation profile (table/cloth background, creases, shadow, blur, JPEG): what training sees |
| `augmentation_by_profile/` | One real CORD and one real SROIE image, original plus three random augmentations each, to show how the profiles differ |
| `app_screens/` | Screenshots of the mobile web app |

Regenerate with `python scripts/make_examples.py`. Real images come from CORD (naver-clova-ix/cord-v2) and SROIE (urbikn/sroie-datasetv2 on Kaggle);
do not redistribute them outside the terms of those datasets.

Contents:
""" + "\n".join("- " + l for l in log) + "\n")
print("\n".join(log)); print("done ->", OUT)
