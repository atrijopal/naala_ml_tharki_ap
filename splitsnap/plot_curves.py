# splitsnap/plot_curves.py
"""plot_curves.py - loss curve + validation curves from metrics.jsonl (the PS asks for loss curves or before/after metrics).
python -m splitsnap.plot_curves --metrics runs/run1/metrics.jsonl [--metrics runs/run1b/metrics.jsonl] --out curves.png
Stages (several files) are concatenated by epoch. Also writes a markdown table next to the PNG."""
import argparse, json
from .metrics_sroie import selection_score


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line); rows[r["epoch"]] = r          # later stages overwrite repeated epochs
    return [rows[k] for k in sorted(rows)]


def series(rows):
    ep, loss = [r["epoch"] for r in rows], [r["train_loss"] for r in rows]
    val = [r for r in rows if "val" in r]
    out = {"epoch": ep, "loss": loss, "val_epoch": [r["epoch"] for r in val], "selection": [selection_score(r["val"]) for r in val]}
    for src, key in (("cord", "ted_acc"), ("sroie", "score"), ("real", "ted_acc"), ("synth", "ted_acc")):
        out[src] = [r["val"][src][key] if src in r["val"] else None for r in val]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", action="append", required=True); ap.add_argument("--out", default="curves.png")
    a = ap.parse_args()
    rows = load(a.metrics); s = series(rows)
    md = ["| epoch | train loss | selection | CORD TED | SROIE score | real TED | synth TED |", "|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.3f}"
    for r in rows:
        if "val" in r:
            i = s["val_epoch"].index(r["epoch"])
            md.append(f"| {r['epoch']} | {r['train_loss']:.3f} | {f(s['selection'][i])} | {f(s['cord'][i])} | {f(s['sroie'][i])} | {f(s['real'][i])} | {f(s['synth'][i])} |")
        else:
            md.append(f"| {r['epoch']} | {r['train_loss']:.3f} | | | | | |")
    open(a.out.rsplit(".", 1)[0] + ".md", "w").write("\n".join(md) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib missing: wrote the markdown table only"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(s["epoch"], s["loss"], marker="o"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].set_title("training loss")
    for k, lab in (("selection", "selection score"), ("cord", "CORD TED"), ("sroie", "SROIE score"), ("real", "real TED"), ("synth", "synthetic TED")):
        pts = [(e, v) for e, v in zip(s["val_epoch"], s[k]) if v is not None]
        if pts:
            ax[1].plot(*zip(*pts), marker="o", label=lab, linewidth=3 if k == "selection" else 1.2)
    ax[1].set_xlabel("epoch"); ax[1].set_title("validation"); ax[1].legend(); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(a.out, dpi=110); print("wrote", a.out)


if __name__ == "__main__":
    main()
