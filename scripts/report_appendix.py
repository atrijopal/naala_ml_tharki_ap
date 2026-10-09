"""Appendix tables for the LaTeX report, generated from the saved result files in results/ (no model is run).
  python scripts/report_appendix.py"""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); D = "report/data/"
def w(name, text): open(D + name, "w").write(text)
def esc(s): return re.sub(r"([&%$#_{}])", r"\\\1", str(s))
def tab(spec, head, rows): return "\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
pc = lambda x, d=1: "n/a" if x is None else f"{100 * x:.{d}f}\\%"
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"

# ---- A: degradation, in full
C = json.load(open("results/degradation.json"))["conditions"]
def lab(c):
    k, v = c["kind"], c["level"]
    if k == "clean": return "Clean"
    if k == "blur": return f"Blur {v:g} px"
    if k == "tilt": return f"Tilt {v:g}$^\\circ$"
    return f"Faded to {100 * v:.0f}\\% contrast"
w("app_deg_acc.tex", tab("@{}lrrrrr@{}", "Condition & Field F1 & Item F1 & Charge F1 & Grand total right & Bills with an error",
    [f"{lab(c)} & {f3(c['field_f1'])} & {f3(c['item_f1'])} & {f3(c['charge_f1'])} & {pc(c['grand_total_acc'], 0)} & {pc(c['bills_with_an_error'], 0)} \\\\" for c in C]))
w("app_deg_conf.tex", tab("@{}lrrrrrrr@{}", "Condition & Wrong fields & Highlighted & Errors caught & Unflagged right & AUROC & Missed-error bills & Retake prompt",
    [f"{lab(c)} & {c['wrong_fields']} of {c['fields']} & {pc(c['flagged_share'])} & {pc(c['errors_caught'], 0)} & {pc(c['unflagged_correct'])} & {f3(c['auroc'])} & {pc(c['bills_with_an_unflagged_error'], 0)} & {pc(c['retake_share'], 0)} \\\\" for c in C]))

# ---- A: straightening
R = json.load(open("results/deskew_check.json"))
w("app_deskew.tex", tab("@{}lllrrrr@{}", "Tilt & Reading & Estimated skew & Field F1 & Item F1 & Charge F1 & Grand total right",
    [f"{r['tilt_deg']}$^\\circ$ & {'straightened' if r['mode'] == 'deskew' else 'as is'} & {r['median_estimated_skew']:g}$^\\circ$ & {f3(r['field_f1'])} & {f3(r['item_f1'])} & {f3(r['charge_f1'])} & {pc(r['grand_total_acc'], 0)} \\\\" for r in R]))
tilts = sorted({r["tilt_deg"] for r in R})
w("app_deskew_asis.dat", "x y\n" + "\n".join(f"{i + 1} {next(r['field_f1'] for r in R if r['tilt_deg'] == t and r['mode'] == 'as is'):.4f}" for i, t in enumerate(tilts)) + "\n")
w("app_deskew_fix.dat", "x y\n" + "\n".join(f"{i + 1} {next(r['field_f1'] for r in R if r['tilt_deg'] == t and r['mode'] == 'deskew'):.4f}" for i, t in enumerate(tilts)) + "\n")
w("app_deskew_labels.tex", "\\def\\deskewlabels{" + ",".join(f"{t}$^\\circ$" for t in tilts) + "}\n")

# ---- B: errors on clean bills
E = json.load(open("results/error_analysis.json")); cats = list(E["error_counts"].items())
half = (len(cats) + 1) // 2
rows = []
for i in range(half):
    a = cats[i]; b = cats[i + half] if i + half < len(cats) else ("", "")
    rows.append(f"{a[0].capitalize()} & {a[1]} & {b[0].capitalize()} & {b[1]} \\\\")
w("app_err_counts.tex", tab("@{}lrlr@{}", "Kind of mistake & Count & Kind of mistake & Count", rows))
RP = json.load(open("results/repair_check.json"))
w("app_repair.tex", tab("@{}lrr@{}", "Rate suggestion on the same 100 bills & Before & After",
    [f"Field F1 & {RP['field_f1_before']:.4f} & {RP['field_f1_after']:.4f} \\\\", f"Bills with every rate right & {RP['bills_with_all_rates_right_before']} of {RP['bills']} & {RP['bills_with_all_rates_right_after']} of {RP['bills']} \\\\",
     f"Suggestions made & \\multicolumn{{2}}{{r}}{{{RP['suggestions_made']}}} \\\\", f"Suggestions that fix a wrong rate & \\multicolumn{{2}}{{r}}{{{RP['suggestions_that_fix_a_wrong_rate']}}} \\\\", f"Correct rates broken & \\multicolumn{{2}}{{r}}{{{RP['correct_rates_broken']}}} \\\\"]))
rx = re.compile(r"^(.*?) misread: read '(.*?)', label '(.*?)' \((.*?)\)$"); lines, hit, tot = [], 0, 0
for e in E["examples"]:
    g = json.load(open("examples/final_run/" + e["bill"].replace(".jpg", ".json")))
    for t in e["issues"]:
        m = rx.match(t)
        if not m: continue
        kind, read, label, item = m.groups(); fld = {"unit price": "unit_price", "line total": "total", "quantity": "qty", "item name": "name"}.get(kind)
        idx = next((i for i, it in enumerate(g["items"]) if it["name"] == item), None)
        flagged = fld is not None and idx is not None and f"items[{idx}].{fld}" in e["flagged_fields"]
        tot += 1; hit += int(flagged)
        if len(lines) < 9: lines.append(f"{esc(e['bill'].replace('.jpg', ''))} & {esc(item)} & {esc(kind)} & {esc(read)} & {esc(label)} & {'yes' if flagged else 'no'} \\\\")
w("app_err_examples.tex", tab("@{}llllrl@{}", "Bill & Item & Field & Read & Label & Highlighted", lines))
w("app_err_summary.tex", f"{hit} of the {tot}")

# ---- C: real SROIE scans
S = json.load(open("results/sroie_scan_check.json")); yn = lambda ok: "yes" if ok else "no"
w("app_sroie_rows.tex", tab("@{}lrrrrrr@{}", "Scan & Label total & Main task items & Company & Date & Address & Total",
    [f"{esc(r['scan'].replace('sroie_', '').replace('.jpg', ''))} & {esc(r['label_total'])} & {r['main_task']['none']['items']} & {yn(r['aux_task']['company']['exact'])} & {yn(r['aux_task']['date']['exact'])} & {yn(r['aux_task']['address']['exact'])} & {yn(r['aux_task']['total']['exact'])} \\\\" for r in S["rows"]]))
V = S["summary"]["main_task_nonempty"]
w("app_sroie_variants.tex", tab("@{}lr@{}", "Image enhancement before reading & Scans with items (of 8)", [f"{esc(k.replace('none', 'none (as is)'))} & {v} \\\\" for k, v in V.items()]))

# ---- D: calibration in full
K = json.load(open("results/calibration.json"))
w("app_calib_agg.tex", tab("@{}lrrrrrr@{}", "Score & AUROC & ECE raw & ECE calibrated & Cut-off & Errors caught & Unflagged right",
    [f"{esc({'p_first': 'first token', 'p_mean': 'mean token', 'p_min': 'minimum token (used)'}[k])} & {f3(v['auroc'])} & {f3(v['ece_raw'])} & {f3(v['ece_calibrated_cv'])} & {v['flag_below']:.4f} & {pc(v['errors_caught'], 1)} & {pc(v['precision_of_unflagged'], 1)} \\\\" for k, v in K["aggregators"].items()]))
w("app_calib_trade.tex", tab("@{}rrrr@{}", "Fields highlighted & Highlight below & Real errors caught & Unflagged correct",
    [f"{pc(t['flagged_share'], 0)} & {t['flag_below']:.4f} & {pc(t['errors_caught'], 0)} & {pc(t['precision_of_unflagged'], 1)} \\\\" for t in K["tradeoff"]]))
w("app_calib_types.tex", tab("@{}lrrr@{}", "Field type & Fields & Accuracy & AUROC",
    [f"{esc(k.replace('_', ' '))} & {v['n']} & {pc(v['accuracy'], 1)} & {f3(v['auroc'])} \\\\" for k, v in sorted(K["by_field_type"].items(), key=lambda kv: -kv[1]["auroc"])]))
print("appendix tables written:", sorted(f for f in os.listdir(D) if f.startswith("app_")))
