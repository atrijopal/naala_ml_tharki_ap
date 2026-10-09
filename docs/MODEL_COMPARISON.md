# Zero-shot model comparison (before any fine-tuning)

Run: Kaggle T4, kernel `atrijopal/splitsnap-models`, raw output in `out/splitsnap-models/`. Same 20 CORD-test + 20 synthetic Indian bills for every model, same prompt for all VLMs, same metrics (`splitsnap/metrics.py`). VLMs ran fp16 with images capped at 1.4M pixels; no bf16 fallback was needed.


> **Correction (2026-10-09): synthetic scores below are understated.** The synthetic generator had a bug: about 29% of bills were printed in whole rupees while their labels kept decimals (e.g. image shows "CGST 32", label says 31.63), so correct readings were scored wrong. This affects every model equally, so rankings are probably still roughly right, but the absolute synthetic numbers (especially charge F1 and totals) are too low. Fixed in `synth.py`; re-run the synthetic comparison before quoting numbers. CORD scores are not affected.

## Zero-shot baselines (same bills, same metrics)

| Model | CORD field F1 | CORD total ok | Synth field F1 | Synth item F1 | Synth charge F1 | Synth total ok | Mean field F1 | Cost |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | 0.73 | 0.80 | 0.92 | 0.93 | 0.79 | 1.00 | 0.83 | 10.64s/bill, 4.6GB, fp16, parse fails 0/40 |
| Qwen3-VL-4B-Instruct | 0.66 | 0.55 | 0.95 | 0.96 | 0.80 | 1.00 | 0.81 | 10.32s/bill, 9.35GB, fp16, parse fails 0/40 |
| Qwen2.5-VL-3B-Instruct | 0.61 | 0.50 | 0.88 | 0.81 | 0.67 | 1.00 | 0.75 | 13.28s/bill, 7.75GB, fp16, parse fails 4/40 |
| Donut CORD-v2 (public, 0.2B) | 0.91 | 1.00 | 0.51 | 0.41 | 0.15 | 0.45 | 0.71 | 0.6s/bill |
| Qwen2-VL-2B-Instruct | 0.45 | 0.60 | 0.68 | 0.51 | 0.34 | 1.00 | 0.56 | 15.17s/bill, 4.77GB, fp16, parse fails 0/40 |

## Image filters (public CORD Donut, one filter at a time)

| Filter | CORD field F1 | CORD total ok | Synth field F1 | Synth total ok |
|---|---|---|---|---|
| none | 0.91 | 1.00 | 0.51 | 0.45 |
| clahe | 0.88 | 1.00 | 0.49 | 0.50 |
| shadow | 0.93 | 1.00 | 0.51 | 0.60 |
| denoise | 0.91 | 0.95 | 0.52 | 0.50 |
| sharpen | 0.92 | 1.00 | 0.46 | 0.50 |
| deskew | 0.90 | 0.95 | 0.46 | 0.45 |
| shadow+clahe | 0.90 | 1.00 | 0.51 | 0.60 |
| deskew+clahe | 0.88 | 0.95 | 0.51 | 0.45 |

## How to read this (important caveats)
- **n = 20 per source.** One bill is 5 points of accuracy. Differences of a few points are noise.
- **The two test sets favour different models for non-model reasons.**
  - Donut-CORD was *trained on CORD's labelling conventions*, so it wins CORD by design. The VLMs also lose points on CORD to Indonesian number formats (e.g. "45.000" = 45000) and CORD's tax/service labels.
  - The VLMs win the synthetic set partly because the prompt was written around our schema and the synthetic bills are clean renders of our own generator. **Synthetic scores say nothing about real Indian bills.**
- So this table shows which model *starts* stronger on each, not which will be best after fine-tuning. The fair test is fine-tuning both on the same data.

## Findings
- **Qwen3-VL-2B is the best zero-shot VLM per cost:** mean field F1 0.83, 0 parse failures, 4.6 GB, 10.6 s/bill. Qwen3-VL-4B is slightly better on synthetic (0.95) but worse on CORD and needs 9.4 GB.
- Qwen2-VL-2B is poor (0.56). Qwen2.5-VL-3B is no better than the Qwen3 models and had 4/40 unparseable outputs.
- **SmolVLM2-2.2B did not run:** its processor needs the `num2words` package (the runner didn't install it). Untested, not a bad result.
- **Speed gap:** Donut 0.6 s/bill vs VLMs 10-15 s/bill on a T4 (about 17x). That matters for a phone-app demo.
- VLM outputs pass the arithmetic check far more often on synthetic bills (75% for Qwen3 vs 0% for Donut-CORD), but only 5-50% on CORD (CORD's own labels are inconsistent 15% of the time).

## Image filters (public CORD Donut, one filter at a time)
No filter is a clear win. Every difference is within 1-3 bills out of 20.
- `shadow` (shadow removal): CORD +0.02 F1, synthetic grand-total-correct 0.45 to 0.60. The only one worth re-testing, on real bills.
- `clahe` alone lowers CORD F1 (0.91 to 0.88). `sharpen` and `deskew` lower synthetic F1 (0.51 to 0.46).
- Verdict: keep all filters off by default. Re-test `shadow` on real validation bills. These filters were tested on the public CORD model, not on our fine-tune, so the result may not transfer.
