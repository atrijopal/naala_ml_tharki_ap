# splitsnap/runguard.py
"""runguard.py - fail-safe helpers for long unattended runs (torch-free so they are unit-tested locally).
Goal: if anything goes badly wrong (NaN loss, divergence, hang, repeated OOM, model learning nothing), STOP with a clear
reason and a non-zero exit instead of burning GPU quota, and never corrupt the last good checkpoint."""
import json, math, os, shutil, sys, threading, time


class Abort(Exception):
    """Raised only for a TOTAL failure that should end the run now (reason is the message). Ordinary wobble never aborts."""


def write_status(path, **kw):
    kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=1, default=str)
    os.replace(tmp, path)


def atomic_replace_dir(tmp, path):
    """Swap a fully written directory `tmp` into `path`. A crash at any point leaves either the old or the new
    directory intact (the old one is parked at path + '.old' until the new one is in place)."""
    old = path + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(path):
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def recover_dir(path):
    """If a previous run died between the two renames, restore the parked copy."""
    old = path + ".old"
    if not os.path.exists(path) and os.path.exists(old):
        os.replace(old, path)
        return True
    return False


class Watchdog:
    """Kills the process (exit code 3) if beat() has not been called for `stall_s` seconds, i.e. a hang."""
    def __init__(self, stall_s, status_path, exit_fn=os._exit):
        self.stall_s, self.status_path, self.exit_fn, self.t = stall_s, status_path, exit_fn, time.time()
        self._stop = False

    def beat(self):
        self.t = time.time()

    def start(self, poll_s=30):
        def loop():
            while not self._stop:
                time.sleep(poll_s)
                if time.time() - self.t > self.stall_s:
                    msg = f"no progress for {self.stall_s / 60:.0f} min (hang)"
                    write_status(self.status_path, status="aborted", reason=msg)
                    print("ABORT:", msg, flush=True); sys.stdout.flush()
                    self.exit_fn(3)
                    return
        threading.Thread(target=loop, daemon=True).start()
        return self

    def stop(self):
        self._stop = True


class LossGuard:
    def __init__(self, nonfinite_checks=5, diverge_factor=10.0, min_first_val=0.2, zero_loss=1e-5, check_epoch=4):
        self.nf_max, self.factor, self.min_first_val, self.zero_loss = nonfinite_checks, diverge_factor, min_first_val, zero_loss
        self.val_from_epoch = check_epoch                    # not named check_epoch: that is a method
        self.nf, self.min_loss, self.first_val_done = 0, None, False

    def check_step(self, loss_value):
        """Call every ~20 steps with a float. Raises after `nonfinite_checks` consecutive NaN/inf values."""
        self.nf = 0 if math.isfinite(loss_value) else self.nf + 1
        if self.nf >= self.nf_max:
            raise Abort(f"loss was NaN/inf in {self.nf} consecutive checks")

    def check_epoch(self, train_loss, epoch):
        if not math.isfinite(train_loss):
            raise Abort(f"epoch {epoch} train loss is {train_loss}")
        if train_loss < self.zero_loss:
            raise Abort(f"epoch {epoch} train loss is ~0 ({train_loss:.2e}): no learning signal (labels masked or collapsed)")
        if self.min_loss is not None and epoch >= 3 and train_loss > self.factor * self.min_loss:
            raise Abort(f"loss diverged at epoch {epoch}: {train_loss:.3f} vs best epoch {self.min_loss:.3f}")
        self.min_loss = train_loss if self.min_loss is None else min(self.min_loss, train_loss)

    def check_resume(self, prev_loss, first_loss, epoch, factor=5.0):
        """After resuming from a checkpoint the first epoch's loss should be near the last epoch before the pause.
        A big jump means the optimizer/schedule/weights were not restored properly."""
        if prev_loss is not None and math.isfinite(prev_loss) and first_loss > factor * prev_loss:
            raise Abort(f"resume looks broken: epoch {epoch} loss {first_loss:.3f} vs {prev_loss:.3f} before the pause")

    def check_val(self, score, epoch):
        """First validation at or after `check_epoch`: an empty prediction already scores about 0.15 (TED), so a score below
        0.2 means the model still produces nothing usable (the stage-1 failure scored 0.14 at every epoch)."""
        if not self.first_val_done and epoch >= self.val_from_epoch:
            self.first_val_done = True
            if score < self.min_first_val:
                raise Abort(f"validation at epoch {epoch}: score {score:.3f} < {self.min_first_val} (an empty output scores ~0.15): the model is not generating usable output")
