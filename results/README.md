# Measured results

Raw outputs behind every number in the report and the README. All were produced by the scripts in `../scripts/` or by the Kaggle runner; nothing here is edited by hand. Numbers on generated bills are not real-bill results.

| File | What it is | Produced by |
|---|---|---|
| `training_logs/stage1_epochs1-8.log`, `stage2_epochs9-16.log` | Per-epoch loss, time, memory and validation scores of the two training sessions | `kaggle/run_kaggle.py full` |
| `training_logs/heldout_evaluation.log`, `heldout_test_predictions.json` | Held-out test evaluation (CORD 95, SROIE 347, generated 100) and the predictions | `kaggle/run_kaggle.py eval` |
| `validation_confidences.json`, `calibration.json` | Per-field confidences on validation bills; the calibration and the flag threshold the app reads | `splitsnap/evaluate.py --conf`, `splitsnap/calibrate.py` |
| `baseline_before_finetuning.json` | Scores before fine-tuning | `kaggle/run_kaggle.py baseline` |
| `before_after.json` | Not fine-tuned vs fine-tuned on 30 unseen generated bills, with example outputs | `scripts/before_after.py` |
| `degradation.json` | Blur, tilt and fading at three levels each, with confidence behaviour (100 unseen generated bills) | `scripts/degradation_study.py` |
| `deskew_check.json` | Effect of straightening tilted bills before reading | `scripts/deskew_check.py` |
| `error_analysis.json` | Error breakdown on the 100 unseen bills | `scripts/error_analysis.py` |
| `repair_check.json` | Effect of the one-tap rate suggestion | `scripts/repair_check.py` |
| `sroie_scan_check.json` | Main and auxiliary task on 8 real SROIE scans, with enhancement retries | `scripts/sroie_scan_check.py` |
| `load_test.json` | Concurrent uploads: throughput, latency, responsiveness | `scripts/load_test.py` |
| `bench_*.json` | Inference benchmarks: fp16, `torch.compile`, int8, FP8, local test set | `scripts/bench_*.py`, `splitsnap/bench_infer.py` |
