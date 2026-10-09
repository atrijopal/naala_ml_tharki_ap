# Inference speed and the zero-risk optimisations: report

Date 2026-10-09. Model: our fine-tuned Donut, epoch-16 checkpoint (`ckpt/b2/runs/run1/best`). Machine: this laptop, **NVIDIA GeForce RTX 4050 Laptop GPU (5.6 GB)**, PyTorch 2.11 + CUDA 12.8, transformers 4.57.6, one bill at a time, greedy decoding, max 512 tokens.
Reproduce: `python scripts/make_local_testset.py`, `python scripts/export_fp16.py ...`, `python scripts/bench_local.py ...`, `python -m splitsnap.bench_infer ...` (raw output in `bench_local.json`, `bench_infer_local.json`).

## 1. Summary

| Item | Tested here | Result | Verdict |
|---|---|---|---|
| **Run in half precision (fp16)** | yes, 20 images | **1.9x faster** than fp32 (median 285 vs 538 ms), half the GPU memory (0.72 vs 1.43 GB), **identical answers on 20/20** | already in use; keep |
| **Cheaper photo preparation** (decode big JPEGs at reduced size) | yes | on 12-megapixel photos, preparation + processor 235 ms -> 57 ms; whole request 426 -> 261 ms (**-39%**); no effect on small images. Answers identical on 19/20 images and on 96/96 fields in a second test, but **one field flipped from right to wrong** in the first test | **low risk, not zero risk.** Opt-in (`FAST_PREP=1`), recommended after a larger real-photo check |
| **Load the model at startup and warm it up** | yes, measured the cost it removes | first request 626 ms vs about 245 ms for later ones: a one-off penalty of about 380 ms (plus 0.5 s model load) that the first user would otherwise pay | no risk; opt-in `PRELOAD=1`, recommended |
| **Drop the per-step "no unknown token" filter** | yes | no speed gain (288 vs 285 ms), identical answers 20/20 | **not worth it; keep the filter** |
| **Half-precision checkpoint** | yes | file **411 MB vs 809 MB**, loads in about half a second, **identical answers 20/20**, same speed | no risk; use it for download and shipping |

## 2. Where the time goes (fp16, RTX 4050, median of 20 images)

| Stage | Time | Share |
|---|---|---|
| Photo preparation (CPU: decode, resize, processor) | 55 ms (up to 237 ms for 12 MP photos) | 20% |
| Image encoder (Swin, 1280x960) | 79 ms | 29% |
| Decoder (token by token, 66 tokens, 2.03 ms/token) | 135 ms | 50% |
| Parsing the output | 0.2 ms | 0% |
| **Total** | **270 ms** (95th percentile about 500 ms) | |

Reading: the decoder and encoder are about equal in importance, preparation matters only for big photos. The 95th percentile is about twice the median because long bills need more tokens. fp32 would take 519 ms (encoder 198, decoder 263).

## 3. Details and numbers

**fp16 vs fp32 (20 images, 3 runs each, median):** all images p50 285 ms vs 538 ms, p95 505 vs 979 ms. Peak GPU memory 0.72 vs 1.43 GB. All 20 outputs were identical.

**Cold start:** model load to GPU 0.5 s. First request 626 ms; the next three 265, 238, 242 ms. A warm-up request at startup (`PRELOAD=1`) moves that cost out of the first user's wait. On the Kaggle T4 the penalty may differ (not measured).

**Cheaper photo preparation:** the current path decodes the full JPEG, shrinks it to 2560 px with LANCZOS, then the processor shrinks it again to 1280x960. The cheaper path asks the JPEG decoder for a reduced-size decode (it keeps both sides at least 1280 px) and lets the processor do the single final resize.
- 12 MP photos (4): prep only 179 -> 21 ms; prep + processor 235 -> 57 ms; whole request 426 -> 261 ms median (p95 462 -> 296 ms). The model input differs by 0.0022 on average (scale -1 to 1).
- Real scans (1-2 MP, 8) and artificial bills (8): no change at all (the path does nothing when the image is already small).
- Accuracy: test A, 20 images: 19 identical; the one different image was a phone-size photo where the **total changed from 4.9 (correct) to 5 (wrong)** and the address got shorter (both versions wrong). Test B, 24 phone-size photos (8 real scans on 3 backgrounds), 96 fields: **69/96 correct for both paths, no field changed**. Together about 1 changed field in 120, and it was a worse one. The test photos are scans pasted onto a table texture, not real phone photos, and the scans are only 8 distinct receipts, so the samples are correlated and small.

**Half-precision checkpoint:** `scripts/export_fp16.py` writes `ckpt/best_fp16` (model 405 MB, 411 MB with tokenizer files). Loading it and running in fp16 gives the same answers as the original on all 20 images at the same speed.

## 4. Not tested
- Real phone photos with labels (none exist yet); the effect of the cheaper path on real accuracy should be checked on a larger set before it is switched on by default.
- The Kaggle T4 (on which the held-out evaluation measured a median of 0.455 s on a different, mostly small-image set, so the two machines' numbers are not comparable).
- End-to-end HTTP time through the web app, CPU-only inference, `torch.compile` / static cache for the decoder (half of the time), TensorRT, lower input resolution (costs accuracy).

## 5. Recommendation
1. Keep fp16 (already used). Ship the 411 MB half-precision checkpoint.
2. Turn on `PRELOAD=1` for any server.
3. Turn on `FAST_PREP=1` for phone photos once checked on a larger set; it is the only change worth a measurable gain (about 165 ms on a 12 MP photo).
4. Leave the token filter on.
5. If more speed is needed later, the decoder (half the time) is the target: try `torch.compile` with a static cache and re-verify that answers are unchanged.

Current typical latency on this laptop for a phone photo: about 430 ms with the defaults, about 260 ms with `FAST_PREP=1`, about 240 ms once warmed up with `PRELOAD=1` as well.
