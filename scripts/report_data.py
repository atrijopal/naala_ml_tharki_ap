"""Write the data files the LaTeX report reads (pgfplots coordinates and small tables) from the real logs and result files.
  python scripts/report_data.py        (re-run after before_after.json / load_test.json change)"""
import ast, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); D = "report/data/"
def w(name, text): open(D + name, "w").write(text)
def coords(pts): return "x y\n" + "\n".join(f"{x} {y:.4f}" for x, y in pts) + "\n"

rows = {}
for f in ("results/training_logs/stage1_epochs1-8.log", "results/training_logs/stage2_epochs9-16.log"):
    for l in open(f, errors="ignore"):
        l = l.strip()
        if l.startswith("{'epoch'"):
            try: d = ast.literal_eval(l)
            except Exception: continue
            if d.get("epoch_sec", 0) > 100: rows[d["epoch"]] = d
assert sorted(rows) == list(range(1, 17))
w("loss.dat", coords([(e, rows[e]["train_loss"]) for e in range(1, 17)]))
val = [(e, rows[e]["val"]) for e in range(1, 17) if "val" in rows[e]]
w("val_cord.dat", coords([(e, v["cord"]["ted_acc"]) for e, v in val]))
w("val_sroie.dat", coords([(e, v["sroie"]["score"]) for e, v in val]))
w("val_synth.dat", coords([(e, v["synth"]["ted_acc"]) for e, v in val]))
w("epoch_minutes.tex", f"{sum(rows[e]['epoch_sec'] for e in range(1,17)) / 16 / 60:.1f}")

c = json.load(open("results/calibration.json"))
w("tradeoff.dat", coords([(t["flagged_share"] * 100, t["errors_caught"] * 100) for t in c["tradeoff"]]))
w("tradeoff_prec.dat", coords([(t["flagged_share"] * 100, t["precision_of_unflagged"] * 100) for t in c["tradeoff"]]))

if os.path.exists("results/before_after.json"):
    r = json.load(open("results/before_after.json")); b, a = r["before"], r["after"]
    keys = [("TED accuracy", "ted_acc"), ("Field F1", "field_f1"), ("Item F1", "item_f1"), ("Charge F1", "charge_f1"), ("Grand total correct", "grand_total_acc"), ("Subtotal correct", "subtotal_acc"), ("Passes arithmetic check", "pred_arith_consistent")]
    body = "\n".join(f"{n} & {b.get(k, 0):.3f} & {a.get(k, 0):.3f} \\\\" for n, k in keys)
    w("ba_tabular.tex", "\\begin{tabular}{@{}lrr@{}}\n\\toprule\nMeasure & Before (not fine-tuned) & After (fine-tuned)\\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    w("ba_bars.tex", "\n".join(f"\\addplot[fill=ash, draw=ink] coordinates {{" + " ".join(f"({n},{b.get(k,0):.3f})" for n, k in keys[:5]) + "};" for _ in [0]) + "\n" +
      "\\addplot[fill=brick, draw=ink] coordinates {" + " ".join(f"({n},{a.get(k,0):.3f})" for n, k in keys[:5]) + "};")
    w("ba_n.tex", str(r["n"]))
if os.path.exists("results/load_test.json"):
    L = json.load(open("results/load_test.json"))["levels"]
    body = "\n".join(f"{x['clients']} & {x['requests']} & {x['bills_per_s']:.2f} & {x['latency_p50_s']:.2f} & {x['latency_p95_s']:.2f} & {x['validate_probe_median_ms']:.0f} & {x['validate_probe_max_ms']:.0f} \\\\" for x in L)
    w("load_tabular.tex", "\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\nClients at once & Requests & Bills per s & Latency p50 (s) & Latency p95 (s) & Validate p50 (ms) & Validate max (ms)\\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    w("load_thr.dat", coords([(x["clients"], x["bills_per_s"]) for x in L])); w("load_p50.dat", coords([(x["clients"], x["latency_p50_s"]) for x in L])); w("load_p95.dat", coords([(x["clients"], x["latency_p95_s"]) for x in L]))
if os.path.exists("results/degradation.json"):
    C = json.load(open("results/degradation.json"))["conditions"]
    def lab(c): return "clean" if c["kind"] == "clean" else f"{c['kind']} {c['level']:g}"
    w("deg_labels.tex", "\\def\\deglabels{" + ",".join(lab(c) for c in C) + "}\n")
    w("deg_f1.dat", "x y\n" + "\n".join(f"{i + 1} {c['field_f1']:.4f}" for i, c in enumerate(C)) + "\n")
    w("deg_caught.dat", "x y\n" + "\n".join(f"{i + 1} {(c['errors_caught'] or 0):.4f}" for i, c in enumerate(C)) + "\n")
    w("deg_retake.dat", "x y\n" + "\n".join(f"{i + 1} {c['retake_share']:.4f}" for i, c in enumerate(C)) + "\n")
print("wrote", sorted(os.listdir(D)))
