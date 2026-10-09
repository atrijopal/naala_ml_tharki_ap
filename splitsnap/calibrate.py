# splitsnap/calibrate.py
"""calibrate.py - R2: does the model's confidence track its real errors, and where should the app flag fields?
Reads the per-field confidences saved by `evaluate.py --conf` (eval_val_conf.json). For each aggregator (first-token / mean /
min token probability) it reports AUROC (does a low score mean a wrong field?), expected calibration error before and after
isotonic calibration (2-fold cross-validation BY BILL, so no bill is calibrated on itself), and at a chosen precision target the
share of fields flagged and the share of real errors caught. Writes calibration.json (flag threshold + isotonic table) and
CALIBRATION.md. python -m splitsnap.calibrate --conf out/splitsnap-eval/eval_val_conf.json --out calibration"""
import argparse, json, os
import numpy as np
from .confidence import auroc, ece, fit_calibrator, reliability_table

AGGS = ("p_first", "p_mean", "p_min")


def load(path):
    rows = []                                                     # (bill index, source, field type, scores..., correct)
    for bi, item in enumerate(json.load(open(path))["predictions"]):
        for f, ok in zip(item["fields"], item["correct"]):
            rows.append((bi, item["source"], f["path"].split(".")[-1], f["p_first"], f["p_mean"], f["p_min"], int(ok)))
    return rows


def threshold_for_precision(scores, correct, target):
    """Largest threshold t such that accepting every field with score >= t keeps precision >= target (flag the rest)."""
    s, c = np.asarray(scores, float), np.asarray(correct, float)
    order = np.argsort(-s); s, c = s[order], c[order]
    prec = np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(prec >= target)[0]
    return float(s[ok[-1]]) if len(ok) else 1.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True); ap.add_argument("--out", default="calibration")
    ap.add_argument("--flag_share", type=float, default=0.08, help="share of fields the app flags (the lowest-scoring ones): the operating point")
    a = ap.parse_args()
    rows = load(a.conf)
    bills = np.array([r[0] for r in rows]); correct = np.array([r[6] for r in rows])
    report = {"n_fields": len(rows), "n_bills": int(len(set(bills))), "field_accuracy": round(float(correct.mean()), 4), "aggregators": {}}
    folds = bills % 2
    best, best_auc = None, -1
    for k, agg in enumerate(AGGS):
        s = np.array([r[3 + k] for r in rows])
        oof = np.zeros_like(s)                                    # out-of-fold calibrated scores
        for f in (0, 1):
            cal = fit_calibrator(s[folds != f], correct[folds != f]); oof[folds == f] = cal.predict(s[folds == f])
        t = float(np.quantile(s, a.flag_share))                  # flag the lowest-scoring `flag_share` of fields
        flagged = s < t; errors = correct == 0
        r = {"auroc": round(float(auroc(s, correct)), 4), "ece_raw": round(float(ece(s, correct)), 4), "ece_calibrated_cv": round(float(ece(oof, correct)), 4),
             "flag_below": round(t, 4), "flagged_share": round(float(flagged.mean()), 4),
             "errors_caught": round(float((flagged & errors).sum() / max(errors.sum(), 1)), 4),
             "precision_of_unflagged": round(float(correct[~flagged].mean()) if (~flagged).any() else 0, 4)}
        report["aggregators"][agg] = r
        if r["auroc"] > best_auc:
            best, best_auc = agg, r["auroc"]
    report["best_aggregator"] = best
    kb = AGGS.index(best); sb = np.array([r[3 + kb] for r in rows]); errs = correct == 0
    report["tradeoff"] = []
    for share in (0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        t = float(np.quantile(sb, share)); fl = sb < t
        report["tradeoff"].append({"flagged_share": share, "flag_below": round(t, 4), "errors_caught": round(float((fl & errs).sum() / max(errs.sum(), 1)), 4),
                                   "precision_of_unflagged": round(float(correct[~fl].mean()), 4), "flagged_per_20_field_bill": round(20 * share, 1)})
    k = AGGS.index(best); s = np.array([r[3 + k] for r in rows])
    cal = fit_calibrator(s, correct)
    xs = np.unique(np.round(np.linspace(0, 1, 101), 2)); report["isotonic_table"] = {"x": xs.tolist(), "y": [round(float(v), 4) for v in cal.predict(xs)]}
    report["reliability_raw"] = reliability_table(s, correct)
    by_type = {}
    for t in sorted({r[2] for r in rows}):
        m = np.array([r[2] == t for r in rows])
        if m.sum() >= 20 and 0 < correct[m].mean() < 1:
            by_type[t] = {"n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4), "auroc": round(float(auroc(s[m], correct[m])), 4)}
    report["by_field_type"] = by_type
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out + ".json", "w"), indent=1)
    b = report["aggregators"][best]
    md = ["# Confidence calibration (R2)", "",
          f"Data: {report['n_fields']} predicted fields from {report['n_bills']} validation bills (CORD + synthetic); {report['field_accuracy']:.1%} of fields are correct.", "",
          "| Aggregator | AUROC | ECE raw | ECE calibrated (2-fold CV) | flag if score below | fields flagged | errors caught | precision of unflagged |", "|---|---|---|---|---|---|---|---|"]
    for g, r in report["aggregators"].items():
        md.append(f"| {g}{' **(best)**' if g == best else ''} | {r['auroc']} | {r['ece_raw']} | {r['ece_calibrated_cv']} | {r['flag_below']} | {r['flagged_share']:.1%} | {r['errors_caught']:.1%} | {r['precision_of_unflagged']:.1%} |")
    md += ["", f"Operating point: flag the lowest-scoring {a.flag_share:.0%} of fields.", "", "Trade-off with the best aggregator (flag more fields = catch more errors, more taps for the user):", "",
           "| fields flagged | flag if score below | real errors caught | unflagged fields correct | flags on a 20-field bill |", "|---|---|---|---|---|"]
    md += [f"| {t['flagged_share']:.0%} | {t['flag_below']} | {t['errors_caught']:.0%} | {t['precision_of_unflagged']:.1%} | {t['flagged_per_20_field_bill']} |" for t in report["tradeoff"]]
    md += ["", f"Reading: AUROC 0.5 would mean confidence says nothing about errors; 1.0 means every wrong field scores below every right one. "
           f"With `{best}` and the flag threshold {b['flag_below']}, the app highlights {b['flagged_share']:.1%} of fields and thereby catches {b['errors_caught']:.1%} of the real errors; "
           f"the fields it does not flag are {b['precision_of_unflagged']:.1%} correct.", "", "By field type: " + ", ".join(f"{k} (n={v['n']}, acc {v['accuracy']:.0%}, AUROC {v['auroc']})" for k, v in by_type.items())]
    open(a.out + ".md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
