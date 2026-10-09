# splitsnap/valsplit.py
"""valsplit.py - which validation bills train.py scores (torch-free, unit-tested)."""
import random


def pick_val(val_all, n):
    """Validation subset. n >= 40: fixed per-source quotas (real bills first); n < 40 (preflight): round-robin so every
    source, including SROIE, is exercised."""
    by = {}
    for r in sorted(val_all, key=lambda r: r["id"]):
        by.setdefault(r["source"], []).append(r)
    for v in by.values():
        random.Random(0).shuffle(v)
    if n >= 40:
        quota = {"real": 20, "sroie": 25, "cord": 25, "synth": 10}
        return [r for s, v in by.items() for r in v[:quota.get(s, 10)]]
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)]; i += 1
    return out[:n]
