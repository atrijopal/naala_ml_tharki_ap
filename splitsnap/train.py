# splitsnap/train.py
"""train.py - Donut fine-tuning on a Kaggle T4 (fp16, grad accumulation, resumable, TED/F1 validation).
Single GPU:   python -m splitsnap.train --manifest data/manifest.jsonl --out /kaggle/working/run1
Two T4s DDP:  accelerate launch --multi_gpu --num_processes 2 -m splitsnap.train ...   (validation is rank-0 only)"""
import argparse, json, math, os, random, shutil, sys, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from .model import build_model, load_model, sanity_check
from .dataset import DonutJSONLDataset
from .infer import generate_json, generate_raw
from .tokens import json2token
from .metrics import evaluate_records
from .schema import TASK_SROIE
from .valsplit import pick_val
from .metrics_sroie import evaluate_sroie, selection_score
from .runguard import Abort, LossGuard, Watchdog, atomic_replace_dir, recover_dir, write_status


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default="/kaggle/working/run1")
    ap.add_argument("--height", type=int, default=1280); ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--bs", type=int, default=1); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--digit_weight", type=float, default=1.0, help="1.0 = plain cross-entropy")
    ap.add_argument("--adam8bit", action="store_true"); ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--val_every", type=int, default=3); ap.add_argument("--val_n", type=int, default=60)
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop cleanly after N hours (0 = off)")
    ap.add_argument("--chunk_epochs", type=int, default=0, help="stop cleanly after N epochs THIS session (0 = off); rerun with --resume")
    ap.add_argument("--stall_minutes", type=float, default=25, help="abort if no progress for this long (hang watchdog; 0 = off)")
    ap.add_argument("--abort_nan_checks", type=int, default=5, help="abort after this many consecutive NaN/inf loss checks (every 20 steps)")
    ap.add_argument("--diverge_factor", type=float, default=10.0, help="abort if an epoch's loss exceeds this x the best epoch loss")
    ap.add_argument("--abort_if_val_below", type=float, default=0.2, help="abort if the first validation at/after epoch --abort_check_epoch scores below this (empty output scores ~0.15)")
    ap.add_argument("--abort_check_epoch", type=int, default=4)
    ap.add_argument("--max_oom", type=int, default=10, help="abort on the (max_oom+1)-th CUDA out-of-memory")
    ap.add_argument("--val_on_train", action="store_true", help="validate on training samples (overfit gate); also prints raw generations")
    ap.add_argument("--no_save", action="store_true", help="do not write checkpoints (overfit gate)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--repeat", default="real=3", help="oversampling, e.g. real=3,sroie=1")
    ap.add_argument("--sources", default="", help="comma list of sources to train on (empty = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=0, help="cap train samples/epoch (smoke tests); 0 = all")
    ap.add_argument("--no_aug", action="store_true", help="disable augmentation (overfit test)")
    ap.add_argument("--use_crop", action="store_true", help="receipt crop+warp in train AND val (must match serving)")
    return ap.parse_args()


def vocab_weights(tok, w):
    v = torch.ones(len(tok))
    if w != 1.0:
        for t, i in tok.get_vocab().items():
            s = t.replace("▁", "")
            if s and all(c in "0123456789.,-" for c in s):
                v[i] = w                                   # numbers cost more than words
    return v


def token_ce(logits, labels):
    """Plain token cross-entropy with OUR alignment: logits[t] is scored against labels[t] (dataset.py already builds
    decoder_input_ids = [start, t1, t2...] and labels = [t1, t2, ..., </s>]). We do NOT use the library's `labels=` loss: in
    transformers 4.57 it is ForCausalLMLoss, which shifts the labels once more and trains 'predict the token after next'
    (low loss, garbage generation: the stage-1 failure)."""
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def weighted_ce(logits, labels, vw):
    V = logits.size(-1)
    ce = F.cross_entropy(logits.float().reshape(-1, V), labels.reshape(-1), ignore_index=-100, reduction="none")
    mask = (labels.reshape(-1) != -100)
    w = vw[labels.reshape(-1).clamp(min=0)]
    return (ce * w * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def validate(model, processor, recs, device, max_length, grad_ckpt, use_crop=False, tick=None):
    model.eval(); model.decoder.config.use_cache = True
    by, sroie = {}, ([], [])
    with torch.autocast("cuda", dtype=torch.float16):
        for r in recs:
            if tick:
                tick()
            pred = generate_json(model, processor, r["image"], device, max_length, use_crop=use_crop, task=r["task"])
            if r["task"] == TASK_SROIE:                              # auxiliary task: company/date/address/total
                sroie[0].append(pred); sroie[1].append(r["gt"])
                continue
            by.setdefault(r["source"], ([], []))
            by[r["source"]][0].append(pred); by[r["source"]][1].append(r["gt"])
    res = {s: evaluate_records(p, g) for s, (p, g) in by.items()}
    if by:
        res["all"] = evaluate_records(sum((v[0] for v in by.values()), []), sum((v[1] for v in by.values()), []))
    if sroie[0]:
        res["sroie"] = evaluate_sroie(*sroie)
    model.train()
    if grad_ckpt:
        model.decoder.config.use_cache = False
    return res


def save(acc, model, processor, path, opt=None, sched=None, epoch=0, best=-1.0):
    """Write to path.tmp, then swap in atomically: a crash mid-save can never corrupt the last good checkpoint."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    m = acc.unwrap_model(model)
    m.save_pretrained(tmp); processor.save_pretrained(tmp)
    if opt is not None:                                     # optimizer state ~1.6 GB: keep ONLY in `last/`
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": best}, f"{tmp}/train_state.pt")
    atomic_replace_dir(tmp, path)


def run(a, status_path, beat):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=a.accum)
    last, best_dir = f"{a.out}/last", f"{a.out}/best"
    recover_dir(last); recover_dir(best_dir)
    resuming = a.resume and os.path.exists(f"{last}/train_state.pt")
    model, processor = load_model(last) if resuming else build_model(a.height, a.width, a.max_length)
    if not resuming:
        sanity_check(model, processor, a.height, a.width)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(); model.decoder.config.use_cache = False
    srcs = [s for s in a.sources.split(",") if s] or None
    repeat = {k: int(v) for k, v in (kv.split("=") for kv in a.repeat.split(",") if kv)}
    train_ds = DonutJSONLDataset(a.manifest, "train", processor, a.max_length, not a.no_aug, srcs, repeat, a.use_crop,
                                 dynamic_pad=(a.bs == 1))
    if a.max_train:
        train_ds.recs = random.Random(1).sample(train_ds.recs, min(a.max_train, len(train_ds.recs)))
    val_recs = pick_val([r for r in map(json.loads, open(a.manifest)) if r["split"] == "val"], a.val_n)
    if a.val_on_train:
        val_recs = random.Random(0).sample(train_ds.recs, min(a.val_n, len(train_ds.recs)))
    print(f"train samples/epoch: {len(train_ds)} | val: {len(val_recs)}", flush=True)
    if a.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.01)
    else:
        opt = None
        if torch.cuda.is_available():
            try:                                                    # fused AdamW needs params already on the GPU
                model.to(acc.device)
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
            except (TypeError, RuntimeError) as e:
                print("fused AdamW unavailable, using the default:", repr(e)[:120], flush=True)
        if opt is None:
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                        drop_last=True, persistent_workers=a.workers > 0)
    total = math.ceil(len(loader) / a.accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup_ratio * total), total)
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)
    state = {"epoch": 0, "best": -1.0}
    if resuming:                                            # load AFTER prepare so state lands on the GPU
        st = torch.load(f"{last}/train_state.pt", map_location="cpu")
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); state.update(epoch=st["epoch"], best=st["best"])
        print("resumed at epoch", state["epoch"], flush=True)
    vw = vocab_weights(processor.tokenizer, a.digit_weight).to(acc.device) if a.digit_weight != 1.0 else None
    guard = LossGuard(a.abort_nan_checks, a.diverge_factor, a.abort_if_val_below, check_epoch=a.abort_check_epoch)
    prev_loss = None
    if resuming and os.path.exists(f"{a.out}/metrics.jsonl"):       # loss of the last epoch before the pause
        lines = [l for l in open(f"{a.out}/metrics.jsonl") if l.strip()]
        prev_loss = json.loads(lines[-1]).get("train_loss") if lines else None
    start_epoch, ooms = state["epoch"], 0
    t0 = time.time(); log = open(f"{a.out}/metrics.jsonl", "a")
    write_status(status_path, status="running", epoch=state["epoch"], epochs=a.epochs, samples_per_epoch=len(train_ds))
    try:
        for epoch in range(state["epoch"], a.epochs):
            model.train(); run_loss, n, te = torch.zeros((), device=acc.device), 0, time.time()
            for step_i, batch in enumerate(loader):
                beat()
                if step_i == 0 and epoch == start_epoch:           # one-off check that our loss alignment is the intended one
                    with torch.no_grad():
                        um = acc.unwrap_model(model); um.eval()                 # eval: no dropout noise in the comparison
                        o1 = um(pixel_values=batch["pixel_values"], decoder_input_ids=batch["decoder_input_ids"])
                        o2 = um(pixel_values=batch["pixel_values"], decoder_input_ids=batch["decoder_input_ids"], labels=batch["labels"])
                        print(f"LOSS CHECK (step 0): our aligned CE {token_ce(o1.logits, batch['labels']).item():.4f} | "
                              f"library loss {o2.loss.item():.4f} (they differ if the library shifts the labels again)", flush=True)
                        um.train()
                try:
                    with acc.accumulate(model):
                        out = model(pixel_values=batch["pixel_values"], decoder_input_ids=batch["decoder_input_ids"])   # no labels=: see token_ce
                        loss = token_ce(out.logits, batch["labels"]) if vw is None else weighted_ce(out.logits, batch["labels"], vw)
                        acc.backward(loss)
                        if acc.sync_gradients:
                            acc.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                except torch.cuda.OutOfMemoryError:
                    ooms += 1
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print(f"CUDA OOM #{ooms}: skipping batch", flush=True)
                    if ooms > a.max_oom:
                        raise Abort(f"CUDA out of memory {ooms} times")
                    continue
                run_loss += loss.detach(); n += 1
                if step_i % 20 == 0:
                    guard.check_step(float(loss.detach()))          # one GPU sync per 20 steps
            if n == 0:
                raise Abort(f"epoch {epoch + 1}: no successful training step")
            rec = {"epoch": epoch + 1, "train_loss": float(run_loss) / n, "epoch_sec": round(time.time() - te),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "lr": sched.get_last_lr()[0]}
            guard.check_epoch(rec["train_loss"], epoch + 1)         # BEFORE any save: NaN weights never overwrite `last`
            if epoch == start_epoch:
                guard.check_resume(prev_loss, rec["train_loss"], epoch + 1)
                if prev_loss is not None:
                    print(f"resume check: loss {prev_loss:.3f} before the pause -> {rec['train_loss']:.3f} now", flush=True)
            if acc.is_main_process:
                if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
                    score = None
                    try:                                            # a broken validation must not kill hours of training
                        res = validate(acc.unwrap_model(model), processor, val_recs, acc.device, a.max_length, a.grad_ckpt, a.use_crop, tick=beat)
                        rec["val"] = res
                        score = selection_score(res)                # mean of CORD TED, SROIE score, real-bill TED (whichever exist)
                    except Exception as e:
                        if a.val_on_train:                          # the overfit gate must really test generation: a crash is a failure
                            raise Abort(f"validation crashed in the overfit gate: {e!r}"[:300])
                        rec["val_error"] = repr(e)[:300]; print("validation failed (training continues):", rec["val_error"], flush=True)
                        model.train()
                        if a.grad_ckpt:
                            model.decoder.config.use_cache = False
                    beat()
                    if a.val_on_train and "val" in rec:             # gate: show what the model really generates next to the target
                        um = acc.unwrap_model(model); um.eval(); um.decoder.config.use_cache = True
                        with torch.autocast("cuda", dtype=torch.float16):
                            for r in val_recs[:3]:
                                print("DEBUG generated:", generate_raw(um, processor, r["image"], acc.device, a.max_length, r["task"])[:500], flush=True)
                                print("DEBUG target   :", json2token(r["gt"])[:500], flush=True)
                        um.train()
                    if score is not None:
                        guard.check_val(score, epoch + 1)           # aborts if the first validation shows nothing was learned
                        if score > state["best"]:
                            state["best"] = score; rec["new_best"] = True
                            if not a.no_save:
                                save(acc, model, processor, best_dir)
                    elif state["best"] < 0:                         # never leave best/ empty
                        state["best"] = 0.0; rec["new_best"] = "placeholder"
                        if not a.no_save:
                            save(acc, model, processor, best_dir)
                if not a.no_save:
                    save(acc, model, processor, last, opt, sched, epoch + 1, state["best"])
                beat()
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
                write_status(status_path, status="running", epoch=epoch + 1, epochs=a.epochs, train_loss=rec["train_loss"],
                             best=state["best"], elapsed_h=round((time.time() - t0) / 3600, 2), ooms=ooms)
            acc.wait_for_everyone()
            if a.chunk_epochs and (epoch + 1 - start_epoch) >= a.chunk_epochs and epoch + 1 < a.epochs:
                print(f"chunk of {a.chunk_epochs} epochs done at epoch {epoch + 1}/{a.epochs} - rerun with --resume", flush=True)
                write_status(status_path, status="chunk_done", epoch=epoch + 1, epochs=a.epochs, best=state["best"],
                             elapsed_h=round((time.time() - t0) / 3600, 2)); return 0
            if a.max_hours and (time.time() - t0) > a.max_hours * 3600:
                print("time budget reached - stopped cleanly; rerun with --resume")
                write_status(status_path, status="time_budget", epoch=epoch + 1, best=state["best"]); return 0
    except KeyboardInterrupt:
        write_status(status_path, status="interrupted", reason="KeyboardInterrupt (last/ keeps the last full epoch)")
        print("interrupted; `last/` still holds the last completed epoch", flush=True); return 130
    write_status(status_path, status="completed", epochs=a.epochs, best=state["best"], elapsed_h=round((time.time() - t0) / 3600, 2))
    return 0


def main():
    """Exit codes: 0 ok, 1 crash, 2 deliberate abort (NaN/diverged/not learning/OOM), 3 hang, 130 interrupted."""
    a = parse(); os.makedirs(a.out, exist_ok=True)
    status_path = f"{a.out}/STATUS.json"
    write_status(status_path, status="starting")
    wd = Watchdog(a.stall_minutes * 60, status_path).start() if a.stall_minutes else None
    try:
        rc = run(a, status_path, wd.beat if wd else (lambda: None))
    except Abort as e:
        write_status(status_path, status="aborted", reason=str(e)); print("ABORT:", e, flush=True); rc = 2
    except Exception as e:
        write_status(status_path, status="crashed", reason=repr(e)[:500], trace=traceback.format_exc()[-1500:])
        traceback.print_exc(); rc = 1
    if wd:
        wd.stop()
    sys.exit(rc)


if __name__ == "__main__":
    main()
