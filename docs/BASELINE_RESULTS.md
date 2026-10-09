# Baseline: scores before fine-tuning

Run: Kaggle T4, kernel `atrijopal/splitsnap-baseline`, raw output in `out/baseline/baseline.json`.
Sample: 30 CORD test bills and 30 synthetic Indian bills (seed 3000). Scored with our own metrics (`splitsnap/metrics.py`).


> **Correction (2026-10-09): synthetic scores below are understated.** The synthetic generator had a bug: about 29% of bills were printed in whole rupees while their labels kept decimals (e.g. image shows "CGST 32", label says 31.63), so correct readings were scored wrong. This affects every model equally, so rankings are probably still roughly right, but the absolute synthetic numbers (especially charge F1 and totals) are too low. Fixed in `synth.py`; re-run the synthetic comparison before quoting numbers. CORD scores are not affected.

## B1: someone else's model, `naver-clova-ix/donut-base-finetuned-cord-v2` (not ours)

| Test set | n | TED | Field F1 | Item F1 | Charge F1 | Grand total correct | Passes arithmetic check |
|---|---|---|---|---|---|---|---|
| CORD test | 30 | 0.92 | 0.90 | 0.83 | 0.94 | 100% | 77% |
| Synthetic Indian bills | 30 | 0.51 | 0.56 | 0.50 | **0.19** | 60% | **0%** |

The model reads Indonesian receipts well and falls apart on Indian-style bills, especially charges (CGST, SGST, service charge, round-off). That is the domain gap fine-tuning has to close. Median inference was 0.66 s per bill. One of the 60 outputs failed to parse and was scored as empty.

## B0: untrained `donut-base` plus our new tokens (the "before" row)

| Test set | n | TED | Field F1 |
|---|---|---|---|
| CORD test | 6 | 0.20 | 0.00 |
| Synthetic | 6 | 0.07 | 0.00 |

As expected, it cannot produce our schema. Only 12 bills, because an untrained model is slow and the result is clear anyway.

## Data facts found along the way
- CORD converted: 968 usable records, 32 skipped. **146 (15%) fail the arithmetic check** (items plus charges don't equal the total). They stay in training, as the guide says.
- Target length in tokens: median 54, 99th percentile 275, maximum 444. **Nothing exceeds 768**, so truncation is not a risk. `--max_length 512` would be safe and cheaper.

## Caveats
- Small samples (30 and 6 per set). Treat these as a rough baseline, not a result.
- The synthetic set measures "can it read my generator", not real Indian bills. **No real bills exist yet**, so there is no honest real-bill baseline.
