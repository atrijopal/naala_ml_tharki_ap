"""Three worked examples for the report, produced by the running app (real model): photo -> reading -> corrections -> split -> explanation.
Writes report/data/walk.tex and report/data/walk.json.   python scripts/walkthroughs.py [--url http://localhost:8765]"""
import glob, json, os, re, urllib.request, uuid, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8765"); a = ap.parse_args()
PEOPLE = ["Riya", "Aman", "Sara"]


def post_file(path):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="b.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/extract", body, {"Content-Type": f"multipart/form-data; boundary={b}"}), timeout=120))


def post_json(route, obj):
    return json.load(urllib.request.urlopen(urllib.request.Request(a.url + route, json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=60))


def pick():
    lab = {f: json.load(open(f)) for f in sorted(glob.glob("examples/final_run/*.json"))}
    want = [lambda t: "DISCOUNT" in t, lambda t: {"CGST", "SGST", "ROUND_OFF"} <= set(t), lambda t: "SERVICE_CHARGE" in t and "DISCOUNT" not in t and ({"GST"} <= set(t) or {"CGST"} <= set(t))]
    chosen = []
    for w in want:
        for f, g in lab.items():
            t = [c["type"] for c in g["charges"]]
            if f not in chosen and 3 <= len(g["items"]) <= 6 and w(t): chosen.append(f); break
    return chosen, lab


def esc(s): return re.sub(r"([&%$#_{}])", r"\\\1", str(s))
def rs(p): return f"{p / 100:.2f}".replace("-", "$-$")
def num(x):
    try: return float(str(x).replace(",", ""))
    except Exception: return None


files, lab = pick(); out, rec = [], []
for k, f in enumerate(files, 1):
    img = f[:-5] + ".jpg"; g = lab[f]; r = post_file(img); bill = r["bill"]; fixes = []
    for i, (it, gi) in enumerate(zip(bill["items"], g["items"])):
        for fld in ("name", "qty", "unit_price", "total"):
            if (num(it[fld]) != num(gi[fld])) if fld != "name" else (it[fld].strip().lower() != gi[fld].strip().lower()):
                fixes.append(f"{esc(gi['name'])}: {fld.replace('_', ' ')} read {esc(it[fld])}, corrected to {esc(gi[fld])}"); it[fld] = str(gi[fld])
    if len(bill["items"]) != len(g["items"]): fixes.append(f"item count differed ({len(bill['items'])} read, {len(g['items'])} on the bill)"); bill["items"] = [{k2: str(v) for k2, v in x.items()} for x in g["items"]]
    for j, (c, gc) in enumerate(zip(bill["charges"], g["charges"])):
        if c["type"] != gc["type"] or num(c["amount"]) != num(gc["amount"]): fixes.append(f"charge {j+1}: read {esc(c['type'])} {esc(c['amount'])}, corrected to {esc(gc['type'])} {esc(gc['amount'])}"); c.update(type=gc["type"], amount=str(gc["amount"]))
    if len(bill["charges"]) != len(g["charges"]): fixes.append("charge lines differed"); bill["charges"] = [{"type": c["type"], "amount": str(c["amount"])} for c in g["charges"]]
    if num(bill["grand_total"]) != num(g["grand_total"]): fixes.append(f"total read {esc(bill['grand_total'])}, corrected to {esc(g['grand_total'])}"); bill["grand_total"] = str(g["grand_total"])
    n = len(bill["items"]); asg = {"0": {p: 1 for p in PEOPLE}}
    if n > 1: asg["1"] = {"Riya": 2, "Aman": 1}
    for i in range(2, n - 1): asg[str(i)] = {PEOPLE[i % 3]: 1}
    if n > 2: asg[str(n - 1)] = {"Aman": 1, "Sara": 1}
    sp = post_json("/split", {"bill": bill, "people": PEOPLE, "assignments": asg}); R = sp["result"]["people"]
    rows = []
    for i, it in enumerate(bill["items"]):
        cells = [next((x["amount"] for x in R[p]["items"] if x["idx"] == i), 0) for p in PEOPLE]
        rows.append(f"{esc(it['name'])} & " + " & ".join(rs(c) for c in cells) + f" & {rs(sum(cells))} \\\\")
    rows.append("\\midrule")
    for j, c in enumerate(bill["charges"]):
        cells = [R[p]["charges"][j]["amount"] for p in PEOPLE]
        rows.append(f"{esc(c['type'] if c['type'] in ('CGST','SGST','IGST','GST','VAT','TAX') else c['type'].replace('_', ' ').title())} & " + " & ".join(rs(x) for x in cells) + f" & {rs(sum(cells))} \\\\")
    rows.append("\\midrule")
    rows.append("Total (Rs) & " + " & ".join(f"\\textbf{{{rs(R[p]['final'])}}}" for p in PEOPLE) + f" & \\textbf{{{rs(sum(R[p]['final'] for p in PEOPLE))}}} \\\\")
    expl = next(x for x in sp["explanation"] if x.startswith("Sara:")).replace("₹", "Rs ")
    expl = expl.split(" Charges and discounts are shared")[0] + " ..." if len(expl) > 360 else expl
    fixtxt = "; ".join(fixes) if fixes else "none needed"
    tex = (f"\\needspace{{8.2cm}}\\noindent\\begin{{minipage}}[t]{{3.1cm}}\\vspace{{0pt}}\\includegraphics[width=3.1cm]{{../{img}}}\\end{{minipage}}\\hfill\n"
           f"\\begin{{minipage}}[t]{{\\dimexpr\\linewidth-3.4cm\\relax}}\\vspace{{0pt}}{{\\footnotesize\\renewcommand{{\\arraystretch}}{{1.0}}\n"
           f"\\textbf{{Bill {k}.}} The model read {len(bill['items'])} items and {len(bill['charges'])} charges. Corrections made on the check screen: {fixtxt}.\\\\[2pt]\n"
           f"\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}\\toprule & Riya & Aman & Sara & Bill\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\\\[2pt]\n"
           f"\\textit{{{esc(expl)}}}}}\\end{{minipage}}\n\\par\\vspace{{5pt}}\n")
    out.append(tex); rec.append({"image": img, "fixes": fixes, "finals": {p: R[p]["final"] for p in PEOPLE}})
    print(os.path.basename(img), "fixes:", fixes, "finals:", {p: R[p]["final"] for p in PEOPLE})
open("report/data/walk.tex", "w").write("\n".join(out[:2]))   # the report shows two; examples/processed_bills has all three; json.dump(rec, open("report/data/walk.json", "w"), indent=1)
