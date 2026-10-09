# splitsnap/compare.py
"""compare.py - merge baseline.json (Donut), vlm/*.json and prep_ablation.json into comparison.md (+ printed table)."""
import argparse, glob, json, os


def row(name, m, extra):
    f = lambda s, k: f"{m[s][k]:.2f}" if s in m else "-"
    avg = sum(m[s]["field_f1"] for s in m) / max(len(m), 1)
    return (avg, f"| {name} | {f('cord','field_f1')} | {f('cord','grand_total_acc')} | {f('synth','field_f1')} | "
                 f"{f('synth','item_f1')} | {f('synth','charge_f1')} | {f('synth','grand_total_acc')} | {avg:.2f} | {extra} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="."); a = ap.parse_args()
    rows, lines = [], []
    b = f"{a.dir}/baseline.json"
    if os.path.exists(b):
        d = json.load(open(b))["B1_public_cord_model"]
        rows.append(row("Donut CORD-v2 (public, 0.2B)", d["metrics"], f"{d['sec_per_bill_p50']}s/bill"))
    for p in sorted(glob.glob(f"{a.dir}/vlm/*.json")):
        d = json.load(open(p))
        rows.append(row(d["model"].split("/")[-1], d["metrics"],
                        f"{d['sec_per_bill_p50']}s/bill, {d['peak_gb']}GB, {d['dtype']}, parse fails {d['parse_failures']}/{d['n_scored']}"))
    lines += ["## Zero-shot baselines (same bills, same metrics)", "",
              "| Model | CORD field F1 | CORD total ok | Synth field F1 | Synth item F1 | Synth charge F1 | Synth total ok | Mean field F1 | Cost |",
              "|---|---|---|---|---|---|---|---|---|"] + [r for _, r in sorted(rows, reverse=True)]
    pa = f"{a.dir}/prep_ablation.json"
    if os.path.exists(pa):
        d = json.load(open(pa))
        lines += ["", "## Image filters (public CORD Donut, one filter at a time)", "",
                  "| Filter | CORD field F1 | CORD total ok | Synth field F1 | Synth total ok |", "|---|---|---|---|---|"]
        for k, v in d.items():
            m = v["metrics"]
            lines.append(f"| {k} | {m['cord']['field_f1']:.2f} | {m['cord']['grand_total_acc']:.2f} | "
                         f"{m['synth']['field_f1']:.2f} | {m['synth']['grand_total_acc']:.2f} |")
    txt = "\n".join(lines); open(f"{a.dir}/comparison.md", "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
