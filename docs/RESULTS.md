# Results: fine-tuned Donut (epoch 16), held-out test splits

Model: `naver-clova-ix/donut-base` fine-tuned by us, full fine-tune, fp16, 1280x960, 16 epochs (about 4.1 h of GPU time in two sessions). Checkpoint: epoch 16, chosen on validation (mean of CORD TED and SROIE score).
Evaluation: Kaggle T4, fp16, greedy decoding, kernel `atrijopal/splitsnap-eval`, raw output `out/splitsnap-eval/log.txt`. Test bills were never trained on and never used to choose the checkpoint. All metrics are our own implementation (`splitsnap/metrics.py`, `metrics_sroie.py`).

## Accuracy on the held-out test splits

| Test set | n | Result |
|---|---|---|
| **CORD test** | 95 | TED **0.930**; field precision / recall / F1 **0.896 / 0.893 / 0.894**; item F1 **0.876**; charge F1 **0.800**; subtotal correct **94.7%**; grand total correct **94.7%**; predictions that pass the arithmetic check 71.6% |
| **SROIE test** | 347 | company exact match **85.3%** (similarity 0.949); date **95.1%**; address **78.4%** (similarity 0.956); total **87.3%**; micro field precision / recall / F1 **0.879 / 0.865 / 0.872** |
| Synthetic test (our generator) | 100 | TED 0.997; field F1 0.985; item F1 0.997; charge F1 0.972; grand total correct 98%; arithmetic-consistent 99% |
| Real Indian bills | **0** | **not measured: none collected yet** |

Validation (used to pick the checkpoint, so slightly optimistic): CORD val (98) TED 0.958; synthetic val (100) TED 0.9975.

## Speed (T4, fp16, one bill at a time, preprocessing + generation + parsing)

| Set | n | median | 95th percentile | max |
|---|---|---|---|---|
| Test (CORD + SROIE + synthetic) | 542 | **0.455 s** | 1.37 s | 3.06 s |
| Validation with per-field confidences | 198 | 0.58 s | 0.95 s | 1.34 s |

## Before vs after

| | CORD field F1 | CORD TED | Synthetic Indian bills, charge F1 | SROIE |
|---|---|---|---|---|
| Untrained `donut-base` + our tokens (n=12) | 0.00 | 0.20 | 0.00 | cannot produce the schema |
| Public CORD-trained Donut (someone else's model; 30 CORD test bills) | 0.90 | 0.92 | 0.15-0.19 (understated: scored before a generator fix) | not trained for it |
| **Ours, epoch 16** (95 CORD test bills) | **0.894** | **0.930** | **0.972** (100 bills) | **F1 0.872** (347 bills) |

## How to read this
- On CORD our model is on par with the public CORD-trained model (field F1 0.894 vs 0.90, TED 0.930 vs 0.92). The two numbers come from different bill sets (95 vs 30), so this is not a clean comparison; charge F1 on CORD (0.800) is below the public model's 0.94 on its sample.
- The gains over the public model are what it could not do: Indian-style charge lines (synthetic charge F1 0.97 vs 0.15-0.19) and the SROIE fields.
- Synthetic scores measure how well the model reads our own generator and say nothing about real Indian bills. **The real-bill test set is still missing and is the most important open item.**
- CORD labels themselves are inconsistent (15% of converted bills fail the arithmetic check), which caps the arithmetic-consistent percentage and charge F1 on CORD.
