"""Kaggle kernel entry point (runs on a T4). scripts/kaggle_run.sh rewrites the MODE line below.

MODE = models   ~60-80 min : zero-shot comparison of 5 open VLMs vs the public CORD Donut + image-filter ablation
MODE = eda      ~10-20 min (CPU kernel, no GPU quota): profile CORD + SROIE (attach SROIE via EXTRA_DATASETS)
MODE = baseline ~30-40 min : untouched-model scores on a small CORD+synthetic sample, then the bench below
MODE = bench    ~15-25 min : measure s/sample, peak GPU memory, inference latency at several settings
MODE = smoke    ~20-30 min : 1 tiny epoch end to end (train -> validate -> checkpoint)
MODE = overfit  ~20-40 min : 16 images, 40 epochs - must memorise them (pipeline correctness proof)
MODE = full     ~6 h (16 epochs; use CHUNK=4 for 4 kernels of ~1.4 h): the real run; outputs runs/run1/{best,last,metrics.jsonl}
"""
import glob, json, os, random, shutil, subprocess, sys, time

MODE = "bench"
ONLY = ""     # models mode: "donut" (Donut baseline + filter ablation) or a comma list of HF model ids; "" = all
CHUNK = 0     # full mode: stop after this many epochs per kernel run (0 = run all); resume with RESUME_KERNEL

IN = "/kaggle/input"
PROJ = "/kaggle/working/proj"
T0 = time.time()


def sh(cmd, check=True, env=None):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=PROJ, check=False, env={**os.environ, **(env or {})})
    if check and r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


def banner(t):
    print(f"\n{'=' * 8} {t}  [{(time.time() - T0) / 60:.1f} min elapsed] {'=' * 8}", flush=True)


# ---- 1. code ---------------------------------------------------------------------------
# Kaggle may flatten the uploaded folder, so find the package dir by its files, not by its name.
def find_pkg():
    for t in glob.glob(f"{IN}/**/train.py", recursive=True) + glob.glob("/kaggle/working/code_unz/**/train.py", recursive=True):
        if "/proj/" in t or "/runs/" in t:                 # a previous kernel's OUTPUT (attached as input) holds a stale code copy: skip it
            continue
        d = os.path.dirname(t)
        if os.path.exists(f"{d}/schema.py") and os.path.exists(f"{d}/__init__.py"):
            return d

pkg = find_pkg()
if not pkg:                                    # uploaded with --dir-mode zip and not auto-extracted
    import zipfile
    for z in glob.glob(f"{IN}/**/splitsnap.zip", recursive=True):
        zipfile.ZipFile(z).extractall("/kaggle/working/code_unz")
    pkg = find_pkg()
if not pkg:
    tree = subprocess.run(f"find {IN} -maxdepth 4 | head -40", shell=True, capture_output=True, text=True).stdout
    raise SystemExit("splitsnap code dataset not found. /kaggle/input tree:\n" + tree)
src_root = pkg
shutil.rmtree(PROJ, ignore_errors=True)
os.makedirs(PROJ)
shutil.copytree(pkg, f"{PROJ}/splitsnap")
os.makedirs(f"{PROJ}/data", exist_ok=True)
banner(f"MODE={MODE}  code from {src_root}")

# ---- 2. environment --------------------------------------------------------------------
if MODE == "eda":
    sh("pip install -q datasets")
    banner("eda")
    sh("python -m splitsnap.eda --out /kaggle/working/eda", env={"TOKENIZERS_PARALLELISM": "false"})
    shutil.rmtree(PROJ, ignore_errors=True)
    banner("DONE")
    raise SystemExit(0)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -V; "
   "python -c \"import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)\"")
sh('pip install -q "transformers>=4.44,<5" datasets accelerate sentencepiece zss nltk scikit-learn bitsandbytes')
sh("python -c \"import transformers,accelerate,datasets;print('transformers',transformers.__version__,"
   "'accelerate',accelerate.__version__,'datasets',datasets.__version__)\"")

# ---- 3. data ---------------------------------------------------------------------------
banner("data")
n_syn = {"baseline": (0, 0, 40), "models": (0, 0, 40), "bench": (60, 20, 10), "smoke": (60, 20, 10), "overfit": (24, 0, 0), "full": (700, 100, 100), "eval": (0, 100, 100), "benchinfer": (0, 0, 100)}[MODE]
sh("python -m splitsnap.convert_cord")
for split, n, seed in zip(("train", "val", "test"), n_syn, (1000, 2000, 3000)):
    if n:
        sh(f"python -m splitsnap.synth --n {n} --out data/synth --seed {seed} --split {split}")
manifests = ["data/cord/manifest.jsonl"] + [f"data/synth/manifest_{s}.jsonl" for s, n in zip(("train", "val", "test"), n_syn) if n]
real = glob.glob(f"{IN}/**/real/manifest.jsonl", recursive=True) + glob.glob(f"{IN}/splitsnap-real*/**/manifest.jsonl", recursive=True)
if real and MODE in ("full", "smoke"):
    base = os.path.dirname(real[0]); recs = [json.loads(l) for l in open(real[0])]
    for r in recs:
        if not os.path.isabs(r["image"]):
            r["image"] = os.path.join(base, r["image"].replace("data/real/", "", 1))
    os.makedirs("/kaggle/working/proj/data/real", exist_ok=True)
    with open(f"{PROJ}/data/real/manifest.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    manifests.append("data/real/manifest.jsonl")
    print(f"real bills: {len(recs)} records from {real[0]}")
else:
    print("NO real bills attached (expected for bench/overfit). Real-bill TED cannot be computed -> best/ picks by overall TED.")
if MODE in ("full", "smoke", "eval", "benchinfer"):  # SROIE must be attached (EXTRA_DATASETS); fail fast if it is not
    sh("python -m splitsnap.convert_sroie --out data/sroie_manifest.jsonl --val_n 60")
    manifests.append("data/sroie_manifest.jsonl")
sh("python -m splitsnap.build_manifest " + " ".join(manifests))

# ---- 4. stage --------------------------------------------------------------------------
banner(f"stage {MODE}")
ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"}
OUT = "/kaggle/working"

if MODE in ("baseline", "bench"):
    if MODE == "baseline":
        sh("python -m splitsnap.baseline --manifest data/manifest.jsonl --n_cord 30 --n_synth 30 "
           f"--out {OUT}/baseline.json", check=False, env=ENV)
    sh("python -c \"import json,numpy as np;from transformers import DonutProcessor;"
       "from splitsnap.model import BASE;from splitsnap.tokens import json2token,collect_added_tokens;"
       "p=DonutProcessor.from_pretrained(BASE);t=p.tokenizer;t.add_tokens(collect_added_tokens(),special_tokens=True);"
       "r=[json.loads(l) for l in open('data/manifest.jsonl')];r=[x for x in r if x['task']=='<s_splitsnap>'];"
       "L=np.array([len(t(json2token(x['gt'])+t.eos_token,add_special_tokens=False).input_ids) for x in r]);"
       "print('TOKLEN pct50/90/95/99/100',np.percentile(L,[50,90,95,99,100]).tolist(),'frac>768',float((L>768).mean()))\"")
    for cfg in ["--height 1280 --width 960 --grad_ckpt",
                "--height 1280 --width 960",
                "--height 1024 --width 768 --grad_ckpt",
                "--height 1280 --width 960 --grad_ckpt --adam8bit"]:
        sh(f"python -m splitsnap.bench --manifest data/manifest.jsonl {cfg}", check=False, env=ENV)

elif MODE == "models":
    # One kernel per unit of work => results are downloadable as soon as THAT kernel ends (Kaggle shows nothing mid-run).
    ALL = ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
           "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"]
    want = [x for x in ONLY.split(",") if x]
    if not want or "donut" in want:
        sh(f"python -m splitsnap.baseline --manifest data/manifest.jsonl --n_cord 20 --n_synth 20 --n_untrained 0 --out {OUT}/baseline.json",
           check=False, env=ENV)
        sh(f"python -m splitsnap.prep_ablation --n_cord 20 --n_synth 20 --out {OUT}/prep_ablation.json", check=False, env=ENV)
    os.makedirs(f"{OUT}/vlm", exist_ok=True)
    for mid in (ALL if not want else [m for m in want if m != "donut"]):
        banner(mid)
        sh(f"timeout 1800 python -m splitsnap.baseline_vlm --model {mid} --n_cord 20 --n_synth 20 --max_minutes 20 "
           f"--out {OUT}/vlm/{mid.split('/')[-1]}.json", check=False, env=ENV)
        sh("rm -rf ~/.cache/huggingface/hub/models--" + mid.replace("/", "--"), check=False)   # keep disk free
    sh(f"python -m splitsnap.compare --dir {OUT}", check=False)

elif MODE == "eval":
    # Held-out evaluation of the best checkpoint of a finished training kernel (attach it with RESUME_KERNEL=<kernel>):
    # test split per source (CORD, SROIE, synthetic) with real fp16 latency, plus val with per-field confidences for calibration.
    cands = glob.glob(f"{IN}/**/run1/best/config.json", recursive=True)
    assert cands, "no run1/best checkpoint found: attach the training kernel with RESUME_KERNEL=<kernel>"
    best = os.path.dirname(cands[0]); print("checkpoint:", best, flush=True)
    sh(f"python -m splitsnap.evaluate --ckpt {best} --manifest data/manifest.jsonl --split test --max_length 512 --out {OUT}/eval_test.json", env=ENV)
    sh(f"python -m splitsnap.evaluate --ckpt {best} --manifest data/manifest.jsonl --split val --sources cord,synth --conf --max_length 512 --out {OUT}/eval_val_conf.json", env=ENV)

elif MODE == "benchinfer":
    # Inference benchmark of the best checkpoint of a finished training kernel: fp32 vs fp16, per-stage time breakdown, memory, accuracy parity.
    cands = glob.glob(f"{IN}/**/run1/best/config.json", recursive=True)
    assert cands, "no run1/best checkpoint found: attach the training kernel with RESUME_KERNEL=<kernel>"
    best = os.path.dirname(cands[0]); print("checkpoint:", best, flush=True)
    sh(f"python -m splitsnap.bench_infer --ckpt {best} --manifest data/manifest.jsonl --n_cord 50 --n_sroie 50 --n_synth 20 --max_length 512 --out {OUT}/bench_infer.json", env=ENV)

elif MODE == "smoke":
    # Rehearsal of the full run: SAME flags as `full` (no grad ckpt, max_length 512, dynamic padding, fused AdamW, SROIE,
    # all augmentation profiles) on 160 samples, plus a stop-after-1-epoch and an in-kernel --resume (guards the 8+8 staging).
    SM = (f"--out {OUT}/runs/smoke --manifest data/manifest.jsonl --height 1280 --width 960 --bs 1 --accum 8 --lr 3e-5 "
          f"--val_every 1 --repeat real=3 --workers 3 --max_length 512 --val_n 8 --max_train 160 --abort_if_val_below 0 --epochs 2")
    sh(f"python -m splitsnap.train {SM} --chunk_epochs 1", env=ENV)
    print("STATUS after part 1:", open(f"{OUT}/runs/smoke/STATUS.json").read(), flush=True)
    sh(f"python -m splitsnap.train {SM} --resume", env=ENV)
    print("STATUS after part 2:", open(f"{OUT}/runs/smoke/STATUS.json").read(), flush=True)
    sh(f"python -m splitsnap.plot_curves --metrics {OUT}/runs/smoke/metrics.jsonl --out {OUT}/smoke_curves.png", check=False)
    shutil.rmtree(f"{OUT}/runs/smoke/best", ignore_errors=True); shutil.rmtree(f"{OUT}/runs/smoke/last", ignore_errors=True)   # keep output small

elif MODE == "overfit":
    recs = [json.loads(l) for l in open(f"{PROJ}/data/manifest.jsonl")]
    tr = [r for r in recs if r["split"] == "train" and r["source"] == "synth"][:16]
    tiny = []
    for r in tr:
        tiny.append(r); v = dict(r); v["id"] += "_v"; v["split"] = "val"; tiny.append(v)
    open(f"{PROJ}/data/tiny.jsonl", "w").write("\n".join(json.dumps(r) for r in tiny))
    sh(f"python -m splitsnap.train --manifest data/tiny.jsonl --out {OUT}/runs/tiny --epochs 40 --val_every 10 "
       f"--val_n 16 --repeat '' --no_aug --lr 5e-5 --workers 3 --grad_ckpt --height 1024 --width 768 --accum 4", env=ENV)

elif MODE == "full":
    resume = "--resume" if os.path.exists(f"{OUT}/runs/run1/last/train_state.pt") else ""
    prev = glob.glob(f"{IN}/**/run1/last/train_state.pt", recursive=True)      # previous kernel output attached as input
    if prev and not resume:
        shutil.copytree(os.path.dirname(os.path.dirname(prev[0])), f"{OUT}/runs/run1", dirs_exist_ok=True); resume = "--resume"
    FLAGS = ("--height 1280 --width 960 --bs 1 --accum 8 --lr 3e-5 --val_every 2 --repeat real=3 --workers 3 --max_length 512")
    # OVERFIT GATE (replaces the old 16-sample preflight, which could not tell "runs" from "learns"; stage 1 of the first
    # full run trained 2.2 h on a mis-aligned loss and scored ~0): train 24 samples, no augmentation, until the model can
    # memorise them, then require that GENERATION reproduces them (selection score >= 0.3; a broken model scores ~0.15).
    # Failure ends the kernel with ERROR in about 10 minutes and prints raw generations vs targets. check=True.
    if resume:
        banner("resuming a previous stage: preflight skipped (the resume loss check guards this stage)")
    else:
        banner("overfit gate")
        GATE = FLAGS.replace("--accum 8", "--accum 2").replace("--lr 3e-5", "--lr 5e-5").replace("--val_every 2", "--val_every 15")
        sh(f"python -m splitsnap.train --manifest data/manifest.jsonl --out {OUT}/gate {GATE} --epochs 15 --max_train 24 "
           f"--val_n 24 --val_on_train --no_aug --no_save --abort_if_val_below 0.3", env=ENV)
        shutil.rmtree(f"{OUT}/gate", ignore_errors=True)
        banner("overfit gate PASSED -> full run")
    rc = sh(f"python -m splitsnap.train --manifest data/manifest.jsonl --out {OUT}/runs/run1 {FLAGS} --epochs 16 "
            f"--val_n 60 --max_hours 6.5 {resume} " + (f"--chunk_epochs {CHUNK}" if CHUNK else ""), check=False, env=ENV)
    st = f"{OUT}/runs/run1/STATUS.json"
    print("\nFINAL STATUS:", open(st).read() if os.path.exists(st) else "(no status file)", flush=True)
    done = os.path.exists(st) and json.load(open(st)).get("status") == "completed"
    if rc == 0 and done:
        ts = f"{OUT}/runs/run1/last/train_state.pt"                                       # optimizer state: only needed to resume
        if os.path.exists(ts):
            os.remove(ts)
    elif rc == 0:
        print("stage finished (status chunk_done/time_budget): optimizer state KEPT for the next stage", flush=True)
    else:
        banner(f"TRAINING STOPPED EARLY (exit code {rc}: 1 crash, 2 deliberate abort, 3 hang). Checkpoints/metrics kept in runs/run1")
        shutil.rmtree(f"{OUT}/proj", ignore_errors=True)
        raise SystemExit(rc)                                                              # kernel shows ERROR, outputs are still downloadable

# keep the output folder small unless we trained for real
shutil.rmtree(f"{OUT}/proj", ignore_errors=True)       # never leave a code copy in the output (the next kernel would pick it up)
banner("DONE")
