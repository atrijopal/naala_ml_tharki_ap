"""Builds the three notebooks in notebooks/ (so they are reproducible and diff-able).  python scripts/make_notebooks.py"""
import nbformat as nbf

def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def code(s): return nbf.v4.new_code_cell(s.strip("\n"))
def save(cells, path):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    nb.nbformat_minor = 5
    nbf.validate(nb); nbf.write(nb, path); print("wrote", path, len(cells), "cells")

REPO = "https://github.com/atrijopal/naala_ml_tharki_ap.git"
COLAB = "https://colab.research.google.com/github/atrijopal/naala_ml_tharki_ap/blob/main/notebooks/"

# ============================================================== 1. training on Kaggle
save([
md(f"""
# SplitSnap 1: fine-tuning Donut on Kaggle

Reproduces the training run behind the model report: `naver-clova-ix/donut-base` fully fine-tuned on CORD, SROIE and generated Indian-style bills.

**Kaggle settings.** Accelerator: GPU T4 x1. Internet: on. Add the input dataset `urbikn/sroie-datasetv2` (SROIE 2019). CORD is downloaded from Hugging Face by the code.

**Time, measured on a T4:** data preparation about 5 min; training 0.42 s per sample, about 14 min per epoch; 8 epochs per session is about 2.2 h and the 16 epochs take two sessions (126 and 119 min of script time). Kaggle sessions are time-limited, which is why the run is split in two stages that share one 16-epoch learning-rate schedule: stage 2 resumes from the optimiser state saved by stage 1.

**What this notebook is.** It runs the same commands as `kaggle/run_kaggle.py`, the script through which the real run was executed, in the same order and with the same flags. The notebook form itself has not been run end to end on Kaggle; the script has.

Run the cells top to bottom. For stage 2, add the stage 1 notebook output as an input and set `STAGE = 2`.
"""),
code(f"""
import glob, json, os, shutil, subprocess, time

STAGE = 1                       # 1: epochs 1-8, starts from donut-base.   2: epochs 9-16, resumes from stage 1
PREVIOUS_OUTPUT = ""            # stage 2 only: folder of the stage 1 output that contains runs/run1 (find it under /kaggle/input)
EPOCHS, EPOCHS_PER_STAGE = 16, 8
REPO = "{REPO}"
PROJ, OUT = "/kaggle/working/proj", "/kaggle/working"
ENV = {{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"}}
T0 = time.time()

def sh(cmd, env=None, check=True):
    print(f"\\n$ {{cmd}}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=PROJ, env={{**os.environ, **(env or {{}})}})
    if check and r.returncode:
        raise SystemExit(f"FAILED ({{r.returncode}}): {{cmd}}")
    return r.returncode
"""),
md("## 1. Code and environment"),
code("""
shutil.rmtree(PROJ, ignore_errors=True)
subprocess.run(f"git clone -q {REPO} {PROJ}", shell=True, check=True)
os.makedirs(f"{PROJ}/data", exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -V")
sh('pip install -q "transformers>=4.44,<5" datasets accelerate sentencepiece zss nltk scikit-learn bitsandbytes')
sh("python -c \\"import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)\\"")
"""),
md("""
## 2. Data

* **CORD** (Indonesian receipts), converted to our schema (items, typed charges, subtotal, grand total).
* **Generated Indian-style bills**: 700 train, 100 validation, 100 test, with fixed seeds. These are whole generated bills, which go beyond light augmentation; the report discloses and separates them.
* **SROIE** (company, date, address, total) as a second task behind its own start token; 60 training scans are held out as validation.
* `build_manifest` joins the sources, assigns the splits and removes duplicate images across splits.
"""),
code("""
sh("python -m splitsnap.convert_cord")
for split, n, seed in (("train", 700, 1000), ("val", 100, 2000), ("test", 100, 3000)):
    sh(f"python -m splitsnap.synth --n {n} --out data/synth --seed {seed} --split {split}")
sh("python -m splitsnap.convert_sroie --out data/sroie_manifest.jsonl --val_n 60")      # finds SROIE2019 under /kaggle/input
sh("python -m splitsnap.build_manifest data/cord/manifest.jsonl data/synth/manifest_train.jsonl "
   "data/synth/manifest_val.jsonl data/synth/manifest_test.jsonl data/sroie_manifest.jsonl")
"""),
md("""
## 3. Overfit gate (stage 1 only)

Trains on 24 samples without augmentation and requires that **generation** reproduces them (selection score at least 0.3; a model that writes nothing scores about 0.15). It fails in about 10 minutes if something is wrong with data, loss or decoding, before the long run starts.
"""),
code("""
FLAGS = "--height 1280 --width 960 --bs 1 --accum 8 --lr 3e-5 --val_every 2 --repeat real=3 --workers 3 --max_length 512"
if STAGE == 1:
    GATE = FLAGS.replace("--accum 8", "--accum 2").replace("--lr 3e-5", "--lr 5e-5").replace("--val_every 2", "--val_every 15")
    sh(f"python -m splitsnap.train --manifest data/manifest.jsonl --out {OUT}/gate {GATE} --epochs 15 --max_train 24 "
       f"--val_n 24 --val_on_train --no_aug --no_save --abort_if_val_below 0.3", env=ENV)
    shutil.rmtree(f"{OUT}/gate", ignore_errors=True)
"""),
md("""
## 4. Training

Full fine-tune, fp16 mixed precision, 1280x960 input, batch 1 with 8-step gradient accumulation (effective batch 8), AdamW with learning rate 3e-5 and a cosine schedule over 16 epochs. Checkpoints are written atomically; the run stops by itself on a NaN loss, a hang or repeated out-of-memory errors. Validation runs every second epoch, and `best/` is chosen by the mean of the CORD and SROIE validation scores.
"""),
code("""
resume = ""
if STAGE == 2:
    shutil.copytree(f"{PREVIOUS_OUTPUT}/runs/run1", f"{OUT}/runs/run1", dirs_exist_ok=True)   # contains last/train_state.pt
    resume = "--resume"
rc = sh(f"python -m splitsnap.train --manifest data/manifest.jsonl --out {OUT}/runs/run1 {FLAGS} --epochs {EPOCHS} "
        f"--val_n 60 --max_hours 6.5 {resume} --chunk_epochs {EPOCHS_PER_STAGE}", env=ENV, check=False)
status = f"{OUT}/runs/run1/STATUS.json"
print("exit code", rc, "| status:", open(status).read() if os.path.exists(status) else "(none)")
print(f"{(time.time() - T0) / 60:.0f} min elapsed")
"""),
md("## 5. Curves and metrics"),
code("""
sh(f"python -m splitsnap.plot_curves --metrics {OUT}/runs/run1/metrics.jsonl --out {OUT}/curves.png", check=False)
from IPython.display import Image, display
if os.path.exists(f"{OUT}/curves.png"):
    display(Image(f"{OUT}/curves.png"))
"""),
md("""
## 6. Held-out evaluation and export (after stage 2 has completed)

Scores the `best/` checkpoint on the test split of each source (CORD, SROIE, generated), with real fp16 latency, and on validation bills with per-field confidences (used for the calibration). Then exports a half-precision copy (about 411 MB) for the app.
"""),
code("""
best = f"{OUT}/runs/run1/best"
if STAGE == 2 and os.path.exists(f"{best}/config.json"):
    sh(f"python -m splitsnap.evaluate --ckpt {best} --manifest data/manifest.jsonl --split test --max_length 512 --out {OUT}/eval_test.json", env=ENV)
    sh(f"python -m splitsnap.evaluate --ckpt {best} --manifest data/manifest.jsonl --split val --sources cord,synth --conf --max_length 512 --out {OUT}/eval_val_conf.json", env=ENV)
    sh(f"python scripts/export_fp16.py --ckpt {best} --out {OUT}/best_fp16")
    ts = f"{OUT}/runs/run1/last/train_state.pt"                  # optimiser state is only needed to resume
    if os.path.exists(ts):
        os.remove(ts)
else:
    print("Evaluation runs after stage 2 (STAGE = 2) has produced runs/run1/best.")
"""),
md("Download `best_fp16/` from the notebook output and place it in `ckpt/best_fp16/` of the repository to run the app (see the README)."),
], "notebooks/01_train_on_kaggle.ipynb")

# ============================================================== 2. inference and split demo
save([
md(f"""
# SplitSnap 2: read a bill, check it, split it

Loads the fine-tuned model, reads one bill photo with per-field confidences, shows what the app would highlight for review, then splits the bill among named people with exact amounts and written explanations. Runs on a laptop, Colab or Kaggle with a GPU (CPU works, slowly).

You need the fine-tuned checkpoint folder (the README says where to put it). Set `CKPT` below, or the `CKPT` environment variable. On Colab, mount Drive or upload the folder first.

[Open in Colab]({COLAB}02_inference_and_split_demo.ipynb)
"""),
code(f"""
import os, sys, json, pathlib, subprocess

REPO = "{REPO}"
if "google.colab" in sys.modules or os.path.exists("/kaggle"):                     # hosted notebook: fetch the code once
    if not os.path.exists("proj"):
        subprocess.run(f"git clone -q {{REPO}} proj && pip install -q 'transformers>=4.44,<5' sentencepiece zss nltk scikit-learn", shell=True, check=True)
    os.chdir("proj")
root = pathlib.Path.cwd()                                                          # local: walk up to the repository root
while not (root / "splitsnap").exists() and root != root.parent:
    root = root.parent
os.chdir(root); sys.path.insert(0, str(root))

CKPT = os.environ.get("CKPT", "ckpt/best_fp16")        # folder with config.json, model.safetensors, tokenizer files
IMAGE = "examples/demo_bills/bill_07.jpg"              # any bill photo; the ten demo bills are generated, not real photographs
print("repository:", root, "| checkpoint exists:", os.path.exists(CKPT))
"""),
md("## 1. Load the model"),
code("""
import torch
from splitsnap.model import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = load_model(CKPT)
model.to(device).eval()
if device == "cuda":
    model.half()
print("device:", device, "| parameters:", round(sum(p.numel() for p in model.parameters()) / 1e6), "M")
"""),
md("## 2. The photo and its quality check\n\nThe app asks for a retake when a photo is too small, blurred, dark, washed out or faint; the check is cheap and runs before the model."),
code("""
from PIL import Image
from splitsnap.preprocess import quality_report

img = Image.open(IMAGE).convert("RGB")
try:
    from IPython.display import display
    display(img.resize((img.width * 600 // img.height, 600)))
except Exception:
    pass
print(quality_report(img))
"""),
md("## 3. Read the bill, with a confidence for every field"),
code("""
import time
from splitsnap.infer import generate_json

t = time.perf_counter()
with torch.inference_mode():
    bill, fields = generate_json(model, processor, img, device, 512, with_conf=True)
print(f"read in {time.perf_counter() - t:.2f} s on {device}\\n")
print(json.dumps(bill, indent=2))
"""),
md("""
## 4. What the correction screen would highlight

A field's confidence is the lowest probability among the tokens that wrote it. The app highlights fields below the calibrated threshold (the lowest-scoring 8% on validation bills). The arithmetic check marks lines that do not add up and, for a misread rate, offers a one-tap fix.
"""),
code("""
from splitsnap.validate import check_arithmetic

threshold = json.load(open("results/calibration.json"))["aggregators"]["p_min"]["flag_below"]
low = [(f["path"], f["value"], round(f["p_min"], 4)) for f in fields if f["p_min"] < threshold]
print(f"threshold {threshold}: {len(low)} of {len(fields)} fields highlighted")
for path, value, p in low:
    print(f"  {path:28s} {value!s:>12}   confidence {p}")
issues = check_arithmetic(bill)
print("\\narithmetic issues:", issues if issues else "none: the bill adds up")
"""),
md("## 5. Split the bill\n\nAmounts are integer paise. Items are divided by share units (a dish shared three ways gives each person one unit); every tax, service charge, discount and rounding line is divided in proportion to each person's pre-tax subtotal, with largest-remainder rounding so the parts add up exactly."),
code("""
from splitsnap.schema import ensure_schema
from splitsnap.split_engine import split_bill
from splitsnap.explain import charge_summary, explain_all

bill = ensure_schema(bill)
people = ["Riya", "Aman", "Sara"]
n = len(bill["items"])
assignments = {0: {p: 1 for p in people}}                    # first item shared three ways
if n > 1: assignments[1] = {"Riya": 2, "Aman": 1}            # second: two portions to Riya, one to Aman
for i in range(2, n - 1): assignments[i] = {people[i % 3]: 1}
if n > 2: assignments[n - 1] = {"Aman": 1, "Sara": 1}        # last shared by two

result = split_bill(bill, people, assignments)
for p in people:
    r = result["people"][p]
    print(f"{p:6s} items {r['pre_tax'] / 100:9.2f}   charges {r['charge_total'] / 100:8.2f}   total {r['final'] / 100:9.2f}")
total = sum(result["people"][p]["final"] for p in people)
print(f"\\nsum of the amounts: {total / 100:.2f}   (bill total as read: {bill['grand_total']})")
print("\\n" + "\\n\\n".join(explain_all(bill, result)))
"""),
md("## 6. Manual override\n\nAny person's final amount can be overridden. The result is flagged as a manual adjustment, and the difference can be spread over the others."),
code("""
from splitsnap.split_engine import apply_overrides, to_paise

adjusted = apply_overrides(result, {"Riya": to_paise(300)}, rebalance=True)
for p in people:
    r = adjusted["people"][p]
    print(f"{p:6s} {r['final'] / 100:9.2f}   {'(adjusted by hand)' if r.get('manual') else ''}")
print("total:", sum(adjusted['people'][p]['final'] for p in people) / 100)
"""),
md("""
## 7. Optional: the same photo before fine-tuning

Downloads `donut-base` (about 800 MB) with our tokens and input size added but not fine-tuned. It has never seen our output format, so it writes the start token and then stops. Set `RUN_BASELINE = True` to try.
"""),
code("""
RUN_BASELINE = False
if RUN_BASELINE:
    from splitsnap.model import build_model
    base, base_processor = build_model(1280, 960, 512)
    base.to(device).eval()
    if device == "cuda":
        base.half()
    from splitsnap.infer import generate_raw
    with torch.inference_mode():
        print(generate_raw(base, base_processor, img, device, 512))
"""),
], "notebooks/02_inference_and_split_demo.ipynb")

# ============================================================== 3. results and figures
save([
md(f"""
# SplitSnap 3: results and figures

Reads the measured results stored in `results/` and redraws the main tables and figures. Nothing is computed by a model here; every number comes from a file produced by the scripts named in `results/README.md`. Numbers on generated bills are not real-bill results.

[Open in Colab]({COLAB}03_results_and_figures.ipynb)
"""),
code(f"""
import ast, json, os, subprocess, sys, pathlib

if "google.colab" in sys.modules or os.path.exists("/kaggle"):
    if not os.path.exists("proj"):
        subprocess.run("git clone -q {REPO} proj", shell=True, check=True)
    os.chdir("proj")
root = pathlib.Path.cwd()
while not (root / "results").exists() and root != root.parent:
    root = root.parent
os.chdir(root)
import matplotlib.pyplot as plt
load = lambda p: json.load(open(p))
print("results folder:", sorted(os.listdir("results"))[:6], "...")
"""),
md("## Training: loss and validation scores"),
code("""
rows = {}
for f in ("results/training_logs/stage1_epochs1-8.log", "results/training_logs/stage2_epochs9-16.log"):
    for line in open(f, errors="ignore"):
        if line.startswith("{'epoch'"):
            d = ast.literal_eval(line.strip())
            if d.get("epoch_sec", 0) > 100:                     # the real run, not the short overfit gate
                rows[d["epoch"]] = d
ep = sorted(rows)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].semilogy(ep, [rows[e]["train_loss"] for e in ep], "o-", color="#a23b2a"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].grid(alpha=.3)
val = [e for e in ep if "val" in rows[e]]
ax[1].plot(val, [rows[e]["val"]["cord"]["ted_acc"] for e in val], "s-", color="black", label="CORD edit-distance score")
ax[1].plot(val, [rows[e]["val"]["sroie"]["score"] for e in val], "o--", color="#a23b2a", label="SROIE field score")
ax[1].plot(val, [rows[e]["val"]["synth"]["ted_acc"] for e in val], "^:", color="#5e6b2d", label="generated bills")
ax[1].set_xlabel("epoch"); ax[1].set_ylim(0.8, 1.0); ax[1].grid(alpha=.3); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
md("## Held-out test results and before/after fine-tuning"),
code("""
m = load("results/heldout_test_predictions.json")["metrics"]
for src, v in m.items():
    print(src, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if not isinstance(x, (dict, list))})
ba = load("results/before_after.json")
print(f"\\nBefore and after fine-tuning on {ba['n']} unseen generated bills")
for k in ("ted_acc", "field_f1", "item_f1", "charge_f1", "grand_total_acc"):
    print(f"  {k:16s} {ba['before'][k]:.3f} -> {ba['after'][k]:.3f}")
"""),
md("## Robustness to poor photos (generated bills, controlled degradation)"),
code("""
C = load("results/degradation.json")["conditions"]
lab = lambda c: "clean" if c["kind"] == "clean" else f"{c['kind']} {c['level']:g}"
x = range(len(C))
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.bar(x, [c["field_f1"] for c in C], color="#5e6b2d", label="field F1")
ax.plot(x, [c["errors_caught"] or 0 for c in C], "o", color="#a23b2a", label="wrong fields the flags catch")
ax.plot(x, [c["retake_share"] for c in C], "s", color="#b07a1e", label="bills the app asks to retake")
ax.set_xticks(list(x)); ax.set_xticklabels([lab(c) for c in C], rotation=30, ha="right"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3); ax.legend(ncol=3, loc="lower left")
plt.tight_layout(); plt.show()
for r in load("results/deskew_check.json"):
    print(f"tilt {r['tilt_deg']:2d} deg  {r['mode']:9s} field F1 {r['field_f1']:.3f}  item F1 {r['item_f1']:.3f}")
"""),
md("## Confidence: what highlighting more fields buys"),
code("""
K = load("results/calibration.json")
t = K["tradeoff"]
plt.figure(figsize=(5.5, 3.4))
plt.plot([100 * r["flagged_share"] for r in t], [100 * r["errors_caught"] for r in t], "o-", color="#a23b2a", label="real errors caught")
plt.plot([100 * r["flagged_share"] for r in t], [100 * r["precision_of_unflagged"] for r in t], "s--", color="#5e6b2d", label="unhighlighted fields correct")
plt.axvline(8, color="black", ls=":"); plt.xlabel("share of fields highlighted, %"); plt.ylabel("%"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()
print({k: (v["auroc"], v["ece_raw"]) for k, v in K["aggregators"].items()})
"""),
md("## Mistakes on clean bills, and the real-scan check"),
code("""
E = load("results/error_analysis.json")
print("bills with a mistake:", E["bills_with_any_error"], "of", E["bills"], "| wrong fields:", E["errors_total"])
print({k: v for k, v in E["error_counts"].items() if v})
print("rate suggestion:", load("results/repair_check.json"))
S = load("results/sroie_scan_check.json")["summary"]
print("\\nreal SROIE scans, main task returns any item (of", S["scans"], "):", S["main_task_nonempty"])
print("auxiliary SROIE task exact matches:", S["aux_task_fields_exact"])
"""),
md("## Speed and capacity"),
code("""
for r in load("results/load_test.json")["levels"]:
    print(f"{r['clients']} clients: {r['bills_per_s']:.2f} bills/s, p50 {r['latency_p50_s']:.2f} s, p95 {r['latency_p95_s']:.2f} s, validate {r['validate_probe_median_ms']:.0f} ms")
"""),
], "notebooks/03_results_and_figures.ipynb")
