# splitsnap/synth.py
"""synth.py - synthetic Indian-style restaurant/canteen bills with EXACT ground truth.
Usage:  python -m splitsnap.synth --n 1500 --out data/synth --seed 1000 --split train
Use different seeds for train/val/test. Synthetic scores say nothing about real bills - always
report real-bill metrics separately."""
import argparse, glob, json, os, random
from decimal import Decimal as D, ROUND_HALF_UP
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .schema import make_schema, TASK_START
from .augment import augment

MENU = [("Paneer Butter Masala", 180, 260), ("Garlic Naan", 30, 60), ("Veg Biryani", 150, 280),
        ("Cold Coffee", 80, 160), ("Masala Dosa", 60, 140), ("Chicken Biryani", 180, 320),
        ("Dal Tadka", 120, 200), ("Butter Roti", 15, 35), ("Jeera Rice", 90, 160),
        ("Gobi Manchurian", 120, 220), ("Veg Thali", 100, 200), ("Idli Sambar", 40, 90),
        ("Tea", 10, 30), ("Filter Coffee", 20, 50), ("Samosa", 15, 40), ("Chole Bhature", 70, 150),
        ("Fresh Lime Soda", 40, 90), ("Mango Lassi", 60, 120), ("Egg Curry", 110, 200),
        ("Paratha", 30, 70), ("Maggi", 40, 80), ("Veg Sandwich", 50, 100), ("Poha", 30, 60),
        ("Chicken Fried Rice", 140, 240), ("Tandoori Roti", 12, 30), ("Mutton Curry", 220, 380)]
PLACES = ["Hotel Sagar", "Udupi Grand", "Annapurna Mess", "Campus Canteen", "Spice Route",
          "Tandoor Junction", "Shree Bhojanalaya", "Hostel Mess Block C", "Biryani Zone", "Cafe Aroma"]
STREETS = ["12, MG Road", "45, Gandhi Nagar", "Plot 7, Sector 14", "Main Gate, IIT Campus", "88, Station Road"]
CITIES = ["Bengaluru - 560001", "Pune - 411001", "Guwahati - 781001", "Chennai - 600001", "Delhi - 110001"]
PRINT = {"CGST": ["CGST", "CGST @{r}%", "C.G.S.T {r}%"], "SGST": ["SGST", "SGST @{r}%", "S.G.S.T {r}%"],
         "SERVICE_CHARGE": ["Service Charge", "S.Charge", "Service Chg {r}%"],
         "DISCOUNT": ["Discount", "Disc.", "Offer Discount"],
         "PACKAGING": ["Packaging", "Pkg Charge", "Parcel Charge"],
         "ROUND_OFF": ["Round Off", "Rounding", "R/Off"], "VAT": ["VAT @{r}%", "VAT"],
         "GST": ["GST @{r}%", "GST"]}
def _find_families():
    """(regular, bold) font pairs that exist on THIS machine (fonts differ between laptop and Kaggle).
    Matplotlib's bundled DejaVu files are the guaranteed fallback."""
    cands = [("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
             ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"),
             ("/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf", "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf"),
             ("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf"),
             ("/usr/share/fonts/truetype/freefont/FreeMono.ttf", "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
             ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
             ("/usr/share/fonts/truetype/lato/Lato-Regular.ttf", "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"),
             ("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf")]
    out = [(r, b) for r, b in cands if os.path.exists(r) and os.path.exists(b)]
    if not out:
        try:
            import matplotlib
            d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
            out = [(f"{d}/DejaVuSansMono.ttf", f"{d}/DejaVuSansMono-Bold.ttf"), (f"{d}/DejaVuSans.ttf", f"{d}/DejaVuSans-Bold.ttf")]
            out = [(r, b) for r, b in out if os.path.exists(r) and os.path.exists(b)]
        except Exception:
            pass
    return out


FAMILIES = _find_families()


def q2(x):
    return D(x).quantize(D("0.01"), rounding=ROUND_HALF_UP)


def get_font(size, bold=False, fam=None):
    if fam is None and FAMILIES:
        fam = FAMILIES[0]
    if fam is not None:
        return ImageFont.truetype(fam[1] if bold else fam[0], size)
    return ImageFont.load_default()


def make_bill(rng):
    """integer mode (1 bill in 3) is printed with whole rupees ("{:.0f}"), so EVERY money value in the label must
    be a whole number too; otherwise the label contains digits that are not on the page (hallucination bait)."""
    integer = rng.random() < 1 / 3
    qc = (lambda x: q2(x).quantize(D("1"), rounding=ROUND_HALF_UP)) if integer else q2
    picks = rng.sample(MENU, rng.randint(2, 9))
    items = []
    for name, lo, hi in picks:
        qty = rng.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]
        unit = D(rng.randrange(lo, hi + 1, 5)) + (D("0.50") if (rng.random() < 0.15 and not integer) else D(0))
        items.append((name, qty, unit, q2(unit * qty)))
    sub = sum((t for *_, t in items), D(0))
    charges = []                                           # (TYPE, amount, rate or None)
    if rng.random() < 0.35:
        r = rng.choice([5, 10]); charges.append(("SERVICE_CHARGE", qc(sub * r / 100), r))
    if rng.random() < 0.10:
        charges.append(("PACKAGING", D(rng.choice([10, 15, 20, 30])), None))
    disc = D(0)
    if rng.random() < 0.25:
        disc = D(rng.choice([10, 20, 25, 30, 50, 100])) if rng.random() < 0.5 else qc(sub * rng.choice([5, 10, 15, 20]) / 100)
        disc = min(disc, sub)
    base = sub - disc
    mode = rng.choices(["cgst_sgst", "gst", "vat", "none"], [60, 10, 10, 20])[0]
    if mode == "cgst_sgst":
        r = rng.choice([D("2.5"), D(6), D(9)]); t = qc(base * r / 100)
        charges += [("CGST", t, r), ("SGST", t, r)]
    elif mode == "gst":
        r = rng.choice([5, 12, 18]); charges.append(("GST", qc(base * r / 100), r))
    elif mode == "vat":
        r = rng.choice([5, 14]); charges.append(("VAT", qc(base * r / 100), r))
    if disc > 0:
        charges.insert(rng.randint(0, len(charges)), ("DISCOUNT", -disc, None))
    total = sub + sum(a for _, a, _ in charges)
    if not integer and rng.random() < 0.45:
        ro = q2(round(total) - total)
        if ro != 0:
            charges.append(("ROUND_OFF", ro, None)); total += ro
    gt = make_schema(items, [(t, a) for t, a, _ in charges], sub, total)
    meta = {"items": items, "charges": charges, "sub": sub, "total": total, "integer": integer}
    return gt, meta


def _wrap(d, text, font, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + [cur]


def render_bill(meta, rng):
    W, fs = rng.choice([560, 640, 720, 800]), rng.choice([22, 24, 26, 28])
    fam = rng.choice(FAMILIES) if FAMILIES else None
    font, bold, small = get_font(fs, False, fam), get_font(fs, True, fam), get_font(int(fs * 0.85), False, fam)
    pad, lh = 18, int(fs * 1.4)
    paper = rng.choice([(255, 255, 255), (250, 248, 238), (244, 241, 228), (238, 238, 238)])
    ink = rng.choice([(15, 15, 15), (30, 30, 30), (70, 70, 70), (110, 110, 110)])  # last two ~ faded thermal
    img = Image.new("RGB", (W, 3600), paper)
    d = ImageDraw.Draw(img)
    y = pad
    rough, jit = rng.random() < 0.6, rng.uniform(0.4, 1.4)

    def put(xy, text, font=None, fill=None):
        """Draw text; in 'rough' mode each character gets a tiny vertical jitter (imperfect printer / dot matrix)."""
        if not rough or len(text) > 60:
            d.text(xy, text, font=font, fill=fill); return
        x, y0 = xy
        for ch in text:
            d.text((x, y0 + rng.uniform(-jit, jit)), ch, font=font, fill=fill); x += d.textlength(ch, font=font)

    def center(t, f):
        nonlocal y
        put(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=ink); y += lh

    def lr(left, right, f=font):
        nonlocal y
        put((pad, y), left, font=f, fill=ink)
        put((W - pad - d.textlength(right, font=f), y), right, font=f, fill=ink); y += lh

    def rule():
        nonlocal y
        d.line([(pad, y + lh // 2), (W - pad, y + lh // 2)], fill=ink, width=1); y += lh

    center(rng.choice(PLACES).upper(), bold)
    center(rng.choice(STREETS), small); center(rng.choice(CITIES), small)
    if rng.random() < 0.8:
        g = "".join(rng.choice("0123456789") for _ in range(2)) + "".join(rng.choice("ABCDEFGHJKLMNP") for _ in range(5)) \
            + "".join(rng.choice("0123456789") for _ in range(4)) + rng.choice("ABCDEFGHJK") + "1Z" + rng.choice("0123456789")
        center(f"GSTIN: {g}", small)
    lr(f"Bill No: {rng.randint(100, 99999)}", f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025 {rng.randint(8, 22):02d}:{rng.randint(0, 59):02d}", small)
    if rng.random() < 0.6:
        lr(rng.choice(["Table", "Counter", "Token"]) + f": {rng.randint(1, 30)}", rng.choice(["Dine In", "Take Away", "Cashier: 02"]), small)
    rule()
    layout = rng.choice(["4col", "3col", "2col"])
    amt_fmt = "{:.0f}" if meta.get("integer") else "{:.2f}"
    qty_x, rate_x = int(W * 0.50), int(W * 0.72)
    hdr = {"4col": ("Item", "Qty", "Rate", "Amt"), "3col": ("Item", "Qty", "", "Amt"), "2col": ("Item", "", "", "Amt")}[layout]
    put((pad, y), hdr[0], font=bold, fill=ink)
    if layout != "2col":
        put((qty_x, y), hdr[1], font=bold, fill=ink)
    if layout == "4col":
        put((rate_x - d.textlength(hdr[2], font=bold), y), hdr[2], font=bold, fill=ink)
    put((W - pad - d.textlength(hdr[3], font=bold), y), hdr[3], font=bold, fill=ink); y += lh
    rule()
    for name, qty, unit, tot in meta["items"]:
        amt = amt_fmt.format(tot)
        maxw = (qty_x - pad - 10) if layout != "2col" else (W - 2 * pad - 110)
        label = name if layout != "2col" else f"{qty} x {name}"
        lines = _wrap(d, label, font, maxw)
        put((pad, y), lines[0], font=font, fill=ink)
        if layout != "2col":
            put((qty_x, y), str(qty), font=font, fill=ink)
        if layout == "4col":
            r = amt_fmt.format(unit)
            put((rate_x - d.textlength(r, font=font), y), r, font=font, fill=ink)
        put((W - pad - d.textlength(amt, font=font), y), amt, font=font, fill=ink); y += lh
        for extra in lines[1:]:
            put((pad, y), extra, font=font, fill=ink); y += lh
    rule()
    lr(rng.choice(["Sub Total", "Subtotal", "Total"]), amt_fmt.format(meta["sub"]))
    for typ, a, r in meta["charges"]:
        lab = rng.choice(PRINT[typ]).format(r=(f"{r}" if r is not None else ""))
        lr(lab, ("-" if a < 0 else "") + amt_fmt.format(abs(a)))
    rule()
    prefix = rng.choice(["", "Rs ", "Rs. "])
    lr(rng.choice(["Grand Total", "Net Payable", "TOTAL", "Bill Amount"]), prefix + "{:.2f}".format(meta["total"]), bold)
    y += lh // 2
    center(rng.choice(["Thank You! Visit Again", "Have a nice day", "*** Thank you ***"]), small)
    return thermal(img.crop((0, 0, W, y + pad)), rng)


def thermal(img, rng):
    """Thermal-print look (65% of bills): head-wear fading across the width/height, row banding, a dropped-out streak,
    soft edges and paper grain. Ink moves toward the paper colour; the label is unchanged (text stays readable)."""
    if rng.random() > 0.65:
        return img
    a = np.array(img).astype(np.float32); h, w = a.shape[:2]
    paper = np.percentile(a.reshape(-1, 3), 95, axis=0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = np.full((h, w), rng.uniform(0.0, 0.3), np.float32)
    if rng.random() < 0.5:
        f += (xx / w) * rng.uniform(0.0, 0.35) if rng.random() < 0.5 else (1 - xx / w) * rng.uniform(0.0, 0.35)
    if rng.random() < 0.5:
        f += rng.uniform(0.0, 0.3) * (yy / h)
    f += 0.05 * np.sin(yy * rng.uniform(0.15, 0.6) + rng.uniform(0, 6))
    for _ in range(rng.choice([0, 0, 1, 2])):
        x0, wd = rng.randint(0, max(w - 12, 1)), rng.randint(2, 9)
        f[:, x0:x0 + wd] += rng.uniform(0.25, 0.55)
    a = a + (paper - a) * np.clip(f, 0, 0.55)[..., None]
    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 0.8))
    a += np.random.RandomState(rng.randrange(1 << 30)).normal(0, rng.uniform(1.5, 4.5), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def compose(paper_img, rng):
    pw, ph = paper_img.size
    m = rng.randint(20, 120)
    bg = rng.choice([(40, 40, 40), (120, 90, 60), (200, 200, 200), (230, 225, 215), (70, 90, 110)])
    canvas = Image.new("RGB", (pw + 2 * m, ph + 2 * m), bg)
    canvas.paste(paper_img, (m, m))
    if rng.random() < 0.7:
        canvas = canvas.rotate(rng.uniform(-4, 4), expand=True, fillcolor=bg, resample=Image.BICUBIC)
    return canvas


def generate(n, out_dir, seed, split):
    rng = random.Random(seed)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    recs = []
    for i in range(n):
        gt, meta = make_bill(rng)
        flat = render_bill(meta, rng)
        # train bills stay flat (augment profile "synth" adds table/crease/shadow/blur/JPEG on the fly, differently each epoch);
        # val/test bills get ONE fixed realistic augmentation so they are not unrealistically clean.
        img = flat if split == "train" else augment(flat, rng, profile="synth")
        path = f"{out_dir}/images/{split}_{i:05d}.jpg"
        img.save(path, quality=92)
        recs.append({"id": f"synth_{split}_{i:05d}", "image": path, "gt": gt, "task": TASK_START,
                     "source": "synth", "split": split})
    with open(f"{out_dir}/manifest_{split}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {n} synthetic bills to {out_dir} ({split})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--split", default="train")
    a = ap.parse_args()
    generate(a.n, a.out, a.seed, a.split)
