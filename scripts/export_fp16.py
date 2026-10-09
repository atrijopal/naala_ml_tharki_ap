"""Export a checkpoint with half-precision weights (about half the file size, same model once run in fp16).
python scripts/export_fp16.py --ckpt ckpt/b2/.../best --out ckpt/best_fp16        Verify with scripts/bench_local.py."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitsnap.model import load_model

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
model, processor = load_model(a.ckpt)
model.half().save_pretrained(a.out, safe_serialization=True); processor.save_pretrained(a.out)
size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 1e6
print(f"wrote {a.out}: {size:.0f} MB")
