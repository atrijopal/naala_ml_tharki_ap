# splitsnap/eda.py
"""eda.py - profile CORD and SROIE BEFORE fine-tuning: label statistics, image characteristics (size, blur,
brightness, contrast, ink fading, tilt), near-duplicate leakage across splits, and what each implies for augmentation.
python -m splitsnap.eda --out eda            (Kaggle: attach the SROIE dataset(s) as kernel inputs)
Writes eda.json, EDA_REPORT.md, eda_hist.png, eda_samples.png. Every section is wrapped so one failure keeps the rest."""
import argparse, collections, glob, json, os, random, re, time, traceback
import numpy as np
from PIL import Image
from .preprocess import load_image, quality_report, estimate_skew
from .schema import parse_money

KEYS = ["min", "p5", "p25", "p50", "p75", "p95", "max", "mean"]


def pct(a):
    a = np.asarray([x for x in a if x is not None], float)
    if not len(a):
        return {}
    return {k: round(float(v), 3) for k, v in zip(KEYS, [a.min(), *np.percentile(a, [5, 25, 50, 75, 95]), a.max(), a.mean()])}


def img_stats(img):
    w, h = img.size
    q = quality_report(img)
    g = np.array(img.convert("L"))
    return {"w": w, "h": h, "mp": round(w * h / 1e6, 3), "aspect": round(h / w, 3), "blur": q["blur_var"],
            "mean": q["mean"], "std": q["std"], "ink": float(np.percentile(g, 5)), "paper": float(np.percentile(g, 95)),
            "skew": estimate_skew(img), "problems": q["problems"]}


def ahash(img):
    """16x16 difference hash (256 bits). The coarse 8x8 average hash called pale receipts with dark text 'identical'."""
    a = np.array(img.convert("L").resize((17, 16), Image.BILINEAR), float)
    return int("".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).ravel()), 2)


def near_dupes(hashes, thr=12):
    """hashes: {split: [(id, ahash)]} -> number of cross-split pairs within `thr` of 256 bits (likely the same image)."""
    splits, n, ex = list(hashes), 0, []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            for ida, ha in hashes[splits[i]]:
                for idb, hb in hashes[splits[j]]:
                    if bin(ha ^ hb).count("1") <= thr:
                        n += 1
                        if len(ex) < 5:
                            ex.append([ida, idb])
    return {"pairs": n, "examples": ex}


def agg_images(rows):
    out = {k: pct([r[k] for r in rows]) for k in ("w", "h", "mp", "aspect", "blur", "mean", "std", "ink", "paper", "skew")}
    c = collections.Counter(p for r in rows for p in r["problems"])
    out["quality_flags"] = {k: round(v / max(len(rows), 1), 3) for k, v in c.items()}
    out["any_flag"] = round(sum(bool(r["problems"]) for r in rows) / max(len(rows), 1), 3)
    out["abs_skew_gt_2deg"] = round(sum(abs(r["skew"]) > 2 for r in rows) / max(len(rows), 1), 3)
    return out


# ---------------------------------------------------------------- CORD
def eda_cord(limit=0):
    from datasets import load_dataset
    from .convert_cord import cord_to_schema
    from .validate import is_consistent
    ds = load_dataset("naver-clova-ix/cord-v2")
    top, sub, tot, menu_keys = (collections.Counter() for _ in range(4))
    res, hashes, imgs_by_split, thumbs = {"splits": {}}, {}, {}, []
    allrows = []
    for split in ds:
        rows, hashes[split] = [], []
        for i, ex in enumerate(ds[split]):
            if limit and i >= limit:
                break
            gt = json.loads(ex["ground_truth"])["gt_parse"]
            top.update(gt.keys())
            st, tt = gt.get("sub_total"), gt.get("total")
            sub.update((st[0] if isinstance(st, list) and st else st or {}).keys() if not isinstance(st, str) else [])
            tot.update((tt[0] if isinstance(tt, list) and tt else tt or {}).keys() if not isinstance(tt, str) else [])
            menu = gt.get("menu", []); menu = [menu] if isinstance(menu, dict) else (menu or [])
            for m in menu:
                menu_keys.update(m.keys())
            sch = cord_to_schema(gt)
            img = ex["image"].convert("RGB")
            r = img_stats(img)
            r.update(n_menu=len(menu), converted=bool(sch and sch["items"]),
                     consistent=(is_consistent(sch) if sch and sch["items"] else None),
                     nested=sum(1 for m in menu if "sub" in m),
                     charge_types=[c["type"] for c in sch["charges"]] if sch else [],
                     total=(float(parse_money(sch["grand_total"])) if sch and parse_money(sch["grand_total"]) is not None else None),
                     qty=[int(i["qty"]) for i in sch["items"]] if sch and sch["items"] else [],
                     item_price=[float(parse_money(i["total"])) for i in sch["items"] if parse_money(i["total"]) is not None] if sch else [])
            rows.append(r); hashes[split].append((f"{split}_{i}", ahash(img)))
            if len(thumbs) < 6 and split == "train":
                thumbs.append(img.copy())
        allrows += rows
        n = max(len(rows), 1)
        res["splits"][split] = {"n": len(rows), "items_per_receipt": pct([r["n_menu"] for r in rows]),
                                "converted_ok": round(sum(r["converted"] for r in rows) / n, 3),
                                "arithmetic_consistent_of_converted": round(sum(bool(r["consistent"]) for r in rows if r["consistent"] is not None) / max(sum(r["consistent"] is not None for r in rows), 1), 3),
                                "images": agg_images(rows)}
        imgs_by_split[split] = rows
    ct = collections.Counter(t for r in allrows for t in r["charge_types"])
    res.update(n_total=len(allrows), top_keys=dict(top.most_common()), sub_total_keys=dict(sub.most_common()),
               total_keys=dict(tot.most_common()), menu_item_keys=dict(menu_keys.most_common()),
               receipts_with_charge_type={k: round(sum(k in r["charge_types"] for r in allrows) / len(allrows), 3) for k in ct},
               qty=pct([q for r in allrows for q in r["qty"]]), qty_gt1=round(float(np.mean([q > 1 for r in allrows for q in r["qty"]] or [0])), 3),
               item_price=pct([p for r in allrows for p in r["item_price"]]), grand_total=pct([r["total"] for r in allrows]),
               nested_menu_items=round(float(np.mean([r["nested"] > 0 for r in allrows])), 3),
               near_duplicates_across_splits=near_dupes(hashes), images_all=agg_images(allrows))
    return res, allrows, thumbs


# ---------------------------------------------------------------- SROIE
DATE_PATS = [("dd/mm/yyyy", r"^\d{1,2}/\d{1,2}/\d{4}$"), ("dd-mm-yyyy", r"^\d{1,2}-\d{1,2}-\d{4}$"),
             ("dd/mm/yy", r"^\d{1,2}/\d{1,2}/\d{2}$"), ("dd-mm-yy", r"^\d{1,2}-\d{1,2}-\d{2}$"),
             ("yyyy-mm-dd", r"^\d{4}-\d{1,2}-\d{1,2}$"), ("dd Mon yyyy", r"^\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4}$"),
             ("dd.mm.yyyy", r"^\d{1,2}\.\d{1,2}\.\d{2,4}$")]


def read_entities(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except Exception:
        d = {}
        for line in t.splitlines():
            if ":" in line:
                k, v = line.split(":", 1); d[k.strip().strip('"').lower()] = v.strip().strip('",')
        return d or None


def sroie_roots(base="/kaggle/input"):
    roots = {os.path.dirname(p) if os.path.basename(p) != "SROIE2019" else p
             for p in glob.glob(f"{base}/**/SROIE2019", recursive=True)}
    return sorted(roots)


def pick_dir(parent, names):
    for n in names:
        p = os.path.join(parent, n)
        if os.path.isdir(p):
            return p


def eda_sroie_root(root, with_images=True, box_limit=400):
    res = {"root": root, "tree": {}, "splits": {}}
    hashes, rows_all, thumbs = {}, [], []
    for sp in sorted(os.listdir(root)):
        d = os.path.join(root, sp)
        if not os.path.isdir(d):
            continue
        sub = sorted(os.listdir(d)); res["tree"][sp] = {x: len(os.listdir(os.path.join(d, x))) for x in sub if os.path.isdir(os.path.join(d, x))}
        imd, end, boxd = pick_dir(d, ["img", "image", "images"]), pick_dir(d, ["entities", "key", "labels"]), pick_dir(d, ["box", "boxes"])
        if not imd:
            continue
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(imd) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        ent, fields, dates, totals, comp_len, addr_len, miss = {}, collections.Counter(), collections.Counter(), [], [], [], 0
        for s in stems:
            ep = os.path.join(end, s + ".txt") if end else None
            e = read_entities(ep) if ep and os.path.exists(ep) else None
            if not e:
                miss += 1; continue
            ent[s] = e
            for k in ("company", "date", "address", "total"):
                if str(e.get(k, "")).strip():
                    fields[k] += 1
            dv = str(e.get("date", "")).strip()
            dates[next((n for n, p in DATE_PATS if re.match(p, dv)), "other" if dv else "empty")] += 1
            m = parse_money(e.get("total")); totals.append(float(m) if m is not None else None)
            comp_len.append(len(str(e.get("company", "")))); addr_len.append(len(str(e.get("address", ""))))
        box_stats = {}
        if boxd:
            lines, hits = [], collections.Counter()
            for s in stems[:box_limit]:
                bp = os.path.join(boxd, s + ".txt")
                if os.path.exists(bp):
                    txt = open(bp, encoding="utf-8", errors="ignore").read()
                    lines.append(txt.count("\n") + 1); up = txt.upper()
                    for kw in ("GST", "SST", "SERVICE", "ROUND", "DISCOUNT", "CHANGE", "CASH", "RM", "TAX"):
                        hits[kw] += kw in up
            n = max(len(lines), 1)
            box_stats = {"box_lines_per_receipt": pct(lines), "receipts_mentioning": {k: round(v / n, 3) for k, v in hits.items()}}
        sp_res = {"n_images": len(stems), "entities_found": len(ent), "entities_missing": miss,
                  "field_present": {k: round(fields[k] / max(len(ent), 1), 3) for k in ("company", "date", "address", "total")},
                  "date_formats": dict(dates), "total_parse_ok": round(sum(t is not None for t in totals) / max(len(totals), 1), 3),
                  "total_value": pct(totals), "company_chars": pct(comp_len), "address_chars": pct(addr_len), **box_stats}
        if with_images and stems:
            rows, hashes[sp] = [], []
            for s in stems:
                ip = next(os.path.join(imd, s + e) for e in (".jpg", ".jpeg", ".png", ".JPG") if os.path.exists(os.path.join(imd, s + e)))
                try:
                    img = load_image(ip)
                except Exception:
                    continue
                r = img_stats(img); rows.append(r); hashes[sp].append((s, ahash(img)))
                if len(thumbs) < 6 and sp.lower().startswith("train"):
                    thumbs.append(img.copy())
            sp_res["images"] = agg_images(rows); rows_all += [dict(r, split=sp) for r in rows]
        sp_res["_ids"] = stems
        res["splits"][sp] = sp_res
    ids = {sp: set(v["_ids"]) for sp, v in res["splits"].items()}
    res["id_overlap_between_splits"] = {f"{a}&{b}": len(ids[a] & ids[b]) for a in ids for b in ids if a < b}
    for v in res["splits"].values():
        v.pop("_ids")
    if with_images:
        res["near_duplicates_across_splits"] = near_dupes(hashes); res["images_all"] = agg_images(rows_all) if rows_all else {}
    return res, rows_all, thumbs


# ---------------------------------------------------------------- report
def row_for(name, a):
    f = lambda k, s="p50": a.get(k, {}).get(s, "-")
    return (f"| {name} | {f('mp')} ({f('mp','p5')}-{f('mp','p95')}) | {f('aspect')} | {f('blur')} ({f('blur','p5')}-{f('blur','p95')}) | "
            f"{f('mean')} | {f('std')} | {f('ink')} / {f('paper')} | {f('skew','p5')} / {f('skew','p95')} | {a.get('abs_skew_gt_2deg','-')} | "
            f"{a.get('any_flag','-')} {a.get('quality_flags','')} |")


def plots(sets, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    panels = [("mp", "megapixels", False), ("aspect", "height / width", False), ("blur", "sharpness (Laplacian var)", True),
              ("mean", "mean brightness", False), ("std", "contrast (std)", False), ("skew", "tilt estimate (deg)", False),
              ("ink", "ink darkness (5th pct; high = faded)", False), ("paper", "paper (95th pct)", False)]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for a, (k, t, lg) in zip(ax.ravel(), panels):
        for name, rows in sets.items():
            v = np.array([r[k] for r in rows], float)
            a.hist(np.log10(np.maximum(v, 1e-3)) if lg else v, bins=40, alpha=.5, label=name, density=True)
        a.set_title(("log10 " if lg else "") + t, fontsize=10)
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(f"{out}/eda_hist.png", dpi=90); plt.close(fig)


def grid(sets, out, h=380):
    rows = []
    for name, imgs in sets.items():
        if imgs:
            row = [im.resize((max(int(im.width * h / im.height), 1), h)) for im in imgs[:6]]
            W = sum(i.width for i in row) + 6 * (len(row) - 1); c = Image.new("RGB", (W, h + 24), "white"); x = 0
            for i in row:
                c.paste(i, (x, 24)); x += i.width + 6
            from PIL import ImageDraw; ImageDraw.Draw(c).text((4, 4), name, fill="black"); rows.append(c)
    if rows:
        W, H = max(r.width for r in rows), sum(r.height for r in rows)
        g = Image.new("RGB", (W, H), "white"); y = 0
        for r in rows:
            g.paste(r, (0, y)); y += r.height
        g.save(f"{out}/eda_samples.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eda"); ap.add_argument("--cord_limit", type=int, default=0)
    ap.add_argument("--sroie_base", default="/kaggle/input"); ap.add_argument("--skip_cord", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R, sets, thumbs, t0 = {}, {}, {}, time.time()
    if not a.skip_cord:
        try:
            R["cord"], rows, thumbs["CORD (train)"] = eda_cord(a.cord_limit); sets["CORD"] = rows
            print(f"CORD done {time.time() - t0:.0f}s", flush=True)
        except Exception:
            R["cord_error"] = traceback.format_exc(); print(R["cord_error"], flush=True)
    roots = sroie_roots(a.sroie_base); R["sroie_roots"] = roots
    for i, root in enumerate(roots):
        try:
            res, rows, th = eda_sroie_root(root, with_images=(i == 0))     # image stats for the first copy only
            R[f"sroie_{i}"] = res
            if i == 0:
                sets["SROIE"] = rows; thumbs["SROIE (train)"] = th
            print(f"SROIE root {i} done {time.time() - t0:.0f}s: {root}", flush=True)
        except Exception:
            R[f"sroie_{i}_error"] = traceback.format_exc(); print(R[f"sroie_{i}_error"], flush=True)
    if not roots:
        R["sroie_error"] = "no SROIE2019 folder found under " + a.sroie_base
    json.dump(R, open(f"{a.out}/eda.json", "w"), indent=1, default=str)
    plots(sets, a.out); grid(thumbs, a.out)
    L = ["# EDA: CORD and SROIE", "", "Image columns are median (5th-95th pct) unless shown otherwise. `blur` = Laplacian variance (higher = sharper); "
         "`ink/paper` = 5th/95th percentile gray (ink near 0 = dark print, higher = faded); `skew` = estimated tilt in degrees (5th / 95th pct).", "",
         "| Set | megapixels | h/w | sharpness | brightness | contrast | ink / paper | skew p5 / p95 | share with \\|tilt\\|>2deg | quality-gate flags |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    if "cord" in R:
        L.append(row_for("CORD all", R["cord"]["images_all"]))
    for i in range(len(roots)):
        r = R.get(f"sroie_{i}", {})
        if r.get("images_all"):
            L.append(row_for(f"SROIE all (copy {i})", r["images_all"]))
    L += ["", "Full numbers (labels, splits, duplicates, date formats, charge frequencies) are in `eda.json`."]
    open(f"{a.out}/EDA_REPORT.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
