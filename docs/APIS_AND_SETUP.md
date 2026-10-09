# SplitSnap: APIs, accounts and training time

## 1. What you need

| # | Service | Needed for | Cost | Status on this machine |
|---|---|---|---|---|
| 1 | **Kaggle API** | Run training on a free T4 GPU, store the code, real bills and checkpoints as private datasets, download SROIE | Free | Token file exists at `~/.kaggle/access_token` (user `atrijopal`). Tested with `kaggle datasets list`, which worked. |
| 2 | **Kaggle account with phone verification** | GPU and internet inside notebooks. Kaggle won't give you a GPU without it. | Free | Not checked. The Kaggle site will tell you. |
| 3 | **Hugging Face Hub** (no API key needed) | Downloads `naver-clova-ix/donut-base` (model) and `naver-clova-ix/cord-v2` (receipt dataset). Both are public. | Free | Reachable. No token needed. |
| 4 | Hugging Face token (optional) | Only if downloads get rate-limited, or if you later deploy the demo as a HF Space | Free | Not set. |
| 5 | PyPI / PyTorch wheels | Python libraries | Free | PyPI is reachable but intermittently slow. |

No paid or closed API is used anywhere in the extraction pipeline, which satisfies requirement R4.

## 2. How to get each one

**Kaggle API token**
1. Sign in at kaggle.com, then go to Settings → API → *Create New Token*.
2. Put it where the CLI looks. The new flow uses `~/.kaggle/access_token`. The older flow uses `~/.kaggle/kaggle.json`, with `chmod 600`.
3. Test with `kaggle datasets list -s sroie`. A table of results means it works.
4. Verify your phone at kaggle.com → Settings → Phone Verification.
5. Weekly GPU quota is about 30 hours, per the problem statement. Check your own number on the Kaggle profile page.

**Hugging Face**: nothing to do. To add a token later: huggingface.co → Settings → Access Tokens → create a *read* token, then `export HF_TOKEN=...`.

**Python environment** (used locally for tests and for the Kaggle CLI):
```bash
uv venv --python 3.12 ~/venvs/naala && source ~/venvs/naala/bin/activate
uv pip install numpy pillow opencv-python-headless zss nltk scikit-learn kaggle
python tests/test_core.py
```
The heavy libraries (torch, transformers, accelerate, datasets, bitsandbytes) are installed inside the Kaggle kernel, not on this laptop.

## 3. Why training runs on Kaggle, not this laptop

This laptop has an RTX 4050 with **6 GB** VRAM. Fine-tuning Donut-base (about 200M parameters) at 1280×960 needs more than that even with every memory trick. A Kaggle T4 has 16 GB. So Kaggle does the training and this machine does the development and tests.

## 4. How long training will take (measured on a Kaggle T4)

Measured by the `baseline` kernel (`out/baseline/`): 24 timed training steps per setting, batch 1, fp16, real CORD data with augmentation.

| Setting | s/sample | Peak GPU memory | Hours per 10,000 sample-passes |
|---|---|---|---|
| 1280×960, gradient checkpointing | 0.58 | 5.1 GB | 1.6 |
| **1280×960, no checkpointing** | **0.46** | 7.2 GB | **1.3** |
| 1024×768, gradient checkpointing | 0.47 | 4.4 GB | 1.3 |
| 1280×960, checkpointing + 8-bit AdamW | 0.59 | 3.9 GB | 1.6 |

A T4 has 16 GB, so 1280×960 fits comfortably without checkpointing. That is the fastest setting that keeps full resolution, so the full run uses it.

Forecast for the real run (about 2,600 samples per epoch: CORD 800 + synthetic 1,500 + real bills ×3):

| Run | Time |
|---|---|
| One epoch | about 20–25 min |
| **8 epochs (default)** | **about 3–4 GPU hours**, including validation |
| 30 epochs (the official CORD recipe) | about 12 GPU hours |
| `smoke` + `overfit` before it | about 10–20 min each |
| Data preparation inside each kernel (install, CORD download, synthetic bills) | about 3 min |

What is still uncertain:
- These numbers come from a short benchmark loop, not from `train.py`. `train.py` uses `accelerate` and has not run on a GPU yet, so its speed may differ a little.
- Data loading is borderline. In two of four configs the loader needed 0.3–0.8 s per sample, which could slow training if the 3 CPU workers fall behind. Watch GPU utilisation during the smoke run.
- Inference latency was **not** measured properly. The model had only 24 training steps, so it generated up to the length cap (3.4 s per bill). The one config that stopped early took 0.23 s. Real latency will come after fine-tuning.
- Real-bill data will make epochs slightly longer, but only by a few percent.

## 5. Order of work

1. `bench`: get real timing and memory numbers, then fix the estimate above.
2. `smoke`, then `overfit`. Don't start `full` until overfit reaches TED ≈ 1.0.
3. Collect and label 60–150 real Indian bills. This is the most important input, and it's you, not an API.
4. `full` training, then evaluation, calibration, the app and the report.
