"""Deliverable: three processed sample bills (raw photo -> extracted JSON -> corrections -> final split with explanation), produced by the running app.
Writes examples/processed_bills/bill_N/{1_photo.jpg, 2_extracted.json, 3_corrections.json, 4_corrected_bill_problem_statement_schema.json, 5_split.json, 6_explanation.txt, README.md}.
  python scripts/make_sample_bills.py [--url http://localhost:8765]"""
import argparse, glob, json, os, shutil, urllib.request, uuid
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8765"); ap.add_argument("--out", default="examples/processed_bills"); a = ap.parse_args()
PEOPLE = ["Riya", "Aman", "Sara"]
THR = json.load(open("results/calibration.json"))["aggregators"]["p_min"]["flag_below"]


def post_file(path):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="bill.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/extract", body, {"Content-Type": f"multipart/form-data; boundary={b}"}), timeout=120))
def post_json(route, obj):
    return json.load(urllib.request.urlopen(urllib.request.Request(a.url + route, json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=60))
num = lambda x: (lambda s: float(s) if s.replace(".", "", 1).replace("-", "", 1).isdigit() else None)(str(x).replace(",", "").strip())
inr = lambda p: f"{p / 100:.2f}"

lab = {f: json.load(open(f)) for f in sorted(glob.glob("examples/final_run/*.json"))}
want = [lambda t: "DISCOUNT" in t, lambda t: {"CGST", "SGST", "ROUND_OFF"} <= set(t), lambda t: "SERVICE_CHARGE" in t and "DISCOUNT" not in t and ("GST" in t or "CGST" in t)]
chosen = []
for w in want:
    for f, g in lab.items():
        if f not in chosen and 3 <= len(g["items"]) <= 6 and w([c["type"] for c in g["charges"]]): chosen.append(f); break

for k, f in enumerate(chosen, 1):
    img = f[:-5] + ".jpg"; g = lab[f]; d = f"{a.out}/bill_{k}"; os.makedirs(d, exist_ok=True); shutil.copy(img, f"{d}/1_photo.jpg")
    r = post_file(img); bill = r["bill"]
    low = [fl["path"] for fl in r["fields"] if fl["p_min"] < THR]
    json.dump({"model_output": r["bill"], "low_confidence_fields_highlighted_in_the_app": low, "arithmetic_issues_reported": r["issues"]}, open(f"{d}/2_extracted.json", "w"), indent=2)
    fixes = []
    for i, (it, gi) in enumerate(zip(bill["items"], g["items"])):
        for fld in ("name", "qty", "unit_price", "total"):
            bad = it[fld].strip().lower() != str(gi[fld]).strip().lower() if fld == "name" else num(it[fld]) != num(gi[fld])
            if bad: fixes.append({"where": f"items[{i}].{fld}", "item": gi["name"], "read": it[fld], "corrected_to": str(gi[fld])}); it[fld] = str(gi[fld])
    for j, (c, gc) in enumerate(zip(bill["charges"], g["charges"])):
        if c["type"] != gc["type"] or num(c["amount"]) != num(gc["amount"]): fixes.append({"where": f"charges[{j}]", "read": c, "corrected_to": gc}); c.update(type=gc["type"], amount=str(gc["amount"]))
    if num(bill["grand_total"]) != num(g["grand_total"]): fixes.append({"where": "grand_total", "read": bill["grand_total"], "corrected_to": str(g["grand_total"])}); bill["grand_total"] = str(g["grand_total"])
    json.dump(fixes, open(f"{d}/3_corrections.json", "w"), indent=2)
    json.dump(post_json("/export", {"bill": bill}), open(f"{d}/4_corrected_bill_problem_statement_schema.json", "w"), indent=2)
    n = len(bill["items"]); asg = {"0": {p: 1 for p in PEOPLE}}
    if n > 1: asg["1"] = {"Riya": 2, "Aman": 1}
    for i in range(2, n - 1): asg[str(i)] = {PEOPLE[i % 3]: 1}
    if n > 2: asg[str(n - 1)] = {"Aman": 1, "Sara": 1}
    sp = post_json("/split", {"bill": bill, "people": PEOPLE, "assignments": asg}); R = sp["result"]["people"]
    json.dump({"assignments_share_units": asg, "result": sp["result"]}, open(f"{d}/5_split.json", "w"), indent=2)
    open(f"{d}/6_explanation.txt", "w").write("\n\n".join(sp["explanation"]) + "\n")
    rows = ["| | " + " | ".join(PEOPLE) + " | Bill |", "|---|" + "---:|" * (len(PEOPLE) + 1)]
    for i, it in enumerate(bill["items"]):
        cells = [next((x["amount"] for x in R[p]["items"] if x["idx"] == i), 0) for p in PEOPLE]; rows.append(f"| {it['name']} | " + " | ".join(inr(c) for c in cells) + f" | {inr(sum(cells))} |")
    for j, c in enumerate(bill["charges"]):
        cells = [R[p]["charges"][j]["amount"] for p in PEOPLE]; rows.append(f"| {c['type'].replace('_', ' ').title() if c['type'] not in ('CGST','SGST','IGST','GST','VAT','TAX') else c['type']} | " + " | ".join(inr(x) for x in cells) + f" | {inr(sum(cells))} |")
    rows.append("| **Total (Rs)** | " + " | ".join(f"**{inr(R[p]['final'])}**" for p in PEOPLE) + f" | **{inr(sum(R[p]['final'] for p in PEOPLE))}** |")
    fixtxt = "\n".join(f"- `{x['where']}`: read `{x.get('read')}`, corrected to `{x.get('corrected_to')}`" for x in fixes) or "- none needed: the reading was correct."
    open(f"{d}/README.md", "w").write(f"""# Processed sample bill {k}

A generated Indian-style bill that was never used for training, validation or testing (`{os.path.basename(img)}` in `examples/final_run`). The photo went through the real app and model.

| Step | File |
|---|---|
| 1. Raw photo | `1_photo.jpg` |
| 2. What the model extracted, with the fields the app highlights for review | `2_extracted.json` |
| 3. Corrections a user makes on the check screen | `3_corrections.json` |
| 4. The corrected bill in the problem statement's JSON schema | `4_corrected_bill_problem_statement_schema.json` |
| 5. The split (who had what, every charge divided per person) | `5_split.json` |
| 6. The plain-language explanation | `6_explanation.txt` |

## Corrections

{fixtxt}

## Final split

Three people share the bill: the first item three ways, the second as two portions for Riya and one for Aman, the last between Aman and Sara, the rest one person each. Amounts are in rupees; every non-item line is divided in proportion to each person's pre-tax subtotal.

""" + "\n".join(rows) + f"""

## Explanation for Sara

> {next(x for x in sp['explanation'] if x.startswith('Sara:')).replace(chr(8377), 'Rs ')}
""")
    print("bill", k, os.path.basename(img), "fixes:", len(fixes), "finals:", {p: R[p]["final"] for p in PEOPLE})
