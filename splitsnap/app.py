# splitsnap/app.py
"""app.py - FastAPI backend + the mobile web UI (splitsnap/static).
  DEMO=1 uvicorn splitsnap.app:app --port 8000                       (whole flow with sample bills, no model needed)
  CKPT=runs/run1/best uvicorn splitsnap.app:app --host 0.0.0.0 --port 8000      (real extraction)
Without DEMO and without CKPT the UI still works: photo extraction answers 503 and the user enters the bill by hand.
The model (torch/transformers) is imported lazily on the first /extract, so the rest runs without them."""
import hashlib, io, json, os, threading
from collections import OrderedDict
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .explain import charge_summary, explain_all
from .preprocess import load_image, quality_report
from .schema import CHARGE_TYPES, TASK_SROIE, ensure_schema, to_ps_schema
from .split_engine import apply_overrides, split_bill, to_paise
from .validate import check_arithmetic

STATIC = Path(__file__).parent / "static"
DEMO = os.environ.get("DEMO", "0") == "1"
CKPT = os.environ.get("CKPT", "")
PRELOAD = os.environ.get("PRELOAD", "0") == "1"           # load the model and run one dummy bill at startup, so the first user does not wait
FAST_PREP = os.environ.get("FAST_PREP", "0") == "1"       # cheaper photo decode/resize (JPEG draft); off until verified on the target machine
AUTO_DESKEW_DEG = float(os.environ.get("AUTO_DESKEW_DEG", "6"))   # straighten photos tilted by at least this many degrees before reading; 0 turns it off
USE_CROP = os.environ.get("USE_CROP", "0") == "1"                # MUST equal the policy used in training
def _calibrated_threshold(default=0.7):
    """Flag threshold from results/calibration.json (splitsnap/calibrate.py): the 8% operating point of the best aggregator.
    CONF_THRESHOLD in the environment overrides it. The UI highlights fields whose p_min is below this."""
    try:
        rep = json.loads((Path(os.environ.get("CALIBRATION", Path(__file__).parent.parent / "results" / "calibration.json"))).read_text())
        if rep.get("best_aggregator") == "p_min":
            return float(next(t["flag_below"] for t in rep["tradeoff"] if abs(t["flagged_share"] - 0.08) < 1e-9))
    except Exception:
        pass
    return default


CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", _calibrated_threshold()))
MAX_UPLOAD = 15 * 1024 * 1024
MAX_ITEMS, MAX_PEOPLE = 200, 40

app = FastAPI(title="SplitSnap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
CACHE, _M = OrderedDict(), {}                                     # image-hash -> result (bounded, oldest dropped first); lazily loaded model
CACHE_MAX = int(os.environ.get("CACHE_MAX", "1000"))
GPU = threading.Lock()                                            # one bill on the model at a time; other requests (edits, splits, pages) are not blocked


# ------------------------------------------------------------------ demo bills (no model needed)
DEMO_BILLS = {
    "clean": {"items": [{"name": "Paneer Butter Masala", "qty": "1", "unit_price": "220", "total": "220"},
                        {"name": "Garlic Naan", "qty": "3", "unit_price": "40", "total": "120"},
                        {"name": "Veg Biryani", "qty": "1", "unit_price": "260", "total": "260"},
                        {"name": "Cold Coffee", "qty": "1", "unit_price": "140", "total": "140"}],
              "charges": [{"type": "CGST", "amount": "18.5"}, {"type": "SGST", "amount": "18.5"},
                          {"type": "SERVICE_CHARGE", "amount": "40"}, {"type": "DISCOUNT", "amount": "-50"}],
              "subtotal": "740", "grand_total": "767"},
    "messy": {"items": [{"name": "Chicken Biryani", "qty": "2", "unit_price": "250", "total": "500"},
                        {"name": "Butter Roti", "qty": "4", "unit_price": "30", "total": "120"},
                        {"name": "Mango Lassi", "qty": "2", "unit_price": "60", "total": "130"},
                        {"name": "Tea", "qty": "1", "unit_price": "20", "total": "20"}],
              "charges": [{"type": "CGST", "amount": "18.5"}, {"type": "SGST", "amount": "18.5"},
                          {"type": "ROUND_OFF", "amount": "0.35"}],
              "subtotal": "770", "grand_total": "808"},
}
DEMO_LOWCONF = {"clean": {"items[1].unit_price": 0.52},
                "messy": {"items[2].total": 0.41, "items[3].name": 0.58, "charges[2].amount": 0.55, "grand_total": 0.63}}


def demo_payload(kind):
    bill = ensure_schema(DEMO_BILLS[kind])
    low = DEMO_LOWCONF[kind]
    fields = []
    for path, value in _paths(bill):
        p = low.get(path, 0.999)            # confident fields sit above the calibrated flag threshold (about 0.99)
        fields.append({"path": path, "value": value, "p_first": p, "p_mean": p, "p_min": p})
    return {"bill": bill, "fields": fields}


def _paths(bill):
    """(path, value) leaves in the same 'items[2].total' form the model's confidence records use."""
    out = []
    for i, it in enumerate(bill["items"]):
        out += [(f"items[{i}].{k}", it[k]) for k in ("name", "qty", "unit_price", "total")]
    for i, c in enumerate(bill["charges"]):
        out += [(f"charges[{i}].type", c["type"]), (f"charges[{i}].amount", c["amount"])]
    return out + [("subtotal", bill["subtotal"]), ("grand_total", bill["grand_total"])]


# ------------------------------------------------------------------ input checking
def clean_bill(raw):
    if not isinstance(raw, dict):
        raise HTTPException(400, "bill must be an object")
    bill = ensure_schema(raw)
    if len(bill["items"]) > MAX_ITEMS or len(bill["charges"]) > MAX_ITEMS:
        raise HTTPException(400, f"too many lines (max {MAX_ITEMS})")
    for c in bill["charges"]:
        c["type"] = c["type"] if c["type"] in CHARGE_TYPES else "OTHER"
    return bill


def clean_people(raw):
    if not isinstance(raw, list):
        raise HTTPException(400, "people must be a list of names")
    names, seen = [], set()
    for n in raw:
        n = str(n).strip()
        if not n:
            continue
        if n.casefold() in seen:
            raise HTTPException(400, f"'{n}' is added twice")
        seen.add(n.casefold()); names.append(n[:40])
    if not names:
        raise HTTPException(400, "add at least one person")
    if len(names) > MAX_PEOPLE:
        raise HTTPException(400, f"too many people (max {MAX_PEOPLE})")
    return names


# ------------------------------------------------------------------ model (lazy)
def _model():
    if "m" not in _M:
        if not CKPT:
            raise HTTPException(503, "No model is set up on this server. Enter the bill by hand, or start the server with CKPT=<checkpoint> (or DEMO=1).")
        import torch
        from .infer import generate_json
        from .model import load_model
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = load_model(CKPT)
        model.to(dev).eval()
        if dev == "cuda":
            model.half()
        _M["m"] = (model, processor, dev, generate_json)
        _warm_up()
    return _M["m"]


def _warm_up():
    """One dummy bill: pays the CUDA start-up, kernel selection and allocator costs before a real request does."""
    from PIL import Image as _Image
    model, processor, dev, gen = _M["m"]
    gen(model, processor, _Image.new("RGB", (900, 1400), (240, 236, 226)), dev, 64, fast_prep=FAST_PREP)


# ------------------------------------------------------------------ routes
@app.on_event("startup")
def _startup():
    if PRELOAD and not DEMO:
        _model()


@app.get("/health")
def health():
    return {"ok": True, "demo": DEMO, "model": bool(CKPT), "loaded": "m" in _M, "conf_threshold": CONF_THRESHOLD, "charge_types": CHARGE_TYPES}


@app.get("/demo")
def demo(kind: str = "clean"):
    if kind not in DEMO_BILLS:
        raise HTTPException(404, "unknown sample")
    p = demo_payload(kind)
    return {"quality": None, "issues": check_arithmetic(p["bill"]), "demo": True, "conf_threshold": CONF_THRESHOLD, **p}


@app.post("/extract")
async def extract(file: UploadFile = File(...), kind: str = "clean"):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "That photo is too large (max 15 MB).")
    key = hashlib.sha1(raw).hexdigest()
    hit = CACHE.get(key)
    if hit is None:
        hit = await run_in_threadpool(_read_photo, raw, kind)      # the model runs in a worker thread, not on the event loop
        CACHE[key] = hit
        while len(CACHE) > CACHE_MAX:
            CACHE.popitem(last=False)
    else:
        CACHE.move_to_end(key)
    return {**hit, "conf_threshold": CONF_THRESHOLD}


def _maybe_deskew(img):
    """Straighten a clearly tilted photo. On 100 generated bills tilted 8 and 15 degrees this raised field F1 from 0.978 to 0.985 and from 0.911 to 0.971,
    and it cost 0.3 points on untilted ones, hence the threshold. Returns (image, degrees straightened or 0)."""
    if AUTO_DESKEW_DEG <= 0:
        return img, 0.0
    from .preprocess import enhance_image, estimate_skew
    skew = estimate_skew(img)
    if abs(skew) >= AUTO_DESKEW_DEG:
        return enhance_image(img, "deskew"), round(float(skew), 1)
    return img, 0.0


def _read_photo(raw, kind):
    try:
        img = load_image(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    quality = quality_report(img)                              # the gate runs on the uncropped photo
    img, straightened = _maybe_deskew(img)
    if straightened:
        quality = {**quality, "straightened_degrees": straightened}
    if DEMO:
        out = {"quality": quality, "demo": True, **demo_payload(kind if kind in DEMO_BILLS else "clean")}
    else:
        with GPU:
            model, processor, dev, generate_json = _model()
            pred, fields = generate_json(model, processor, img, dev, 512, with_conf=True, use_crop=USE_CROP, fast_prep=FAST_PREP)
            partial = None
            if not pred.get("items"):                          # nothing read as a restaurant bill: the receipt-header task can still give the shop and the total
                aux = generate_json(model, processor, img, dev, 512, task=TASK_SROIE)
                if isinstance(aux, dict) and (aux.get("company") or aux.get("total")):
                    partial = {k: str(aux.get(k) or "") for k in ("company", "date", "address", "total")}
        out = {"quality": quality, "demo": False, "bill": pred, "fields": fields}
        if partial:
            out["partial"] = partial
    out["issues"] = check_arithmetic(out["bill"])
    return out


@app.post("/validate")                                      # call after every edit in the correction UI
def validate(payload: dict):
    bill = clean_bill(payload.get("bill"))
    return {"bill": bill, "issues": check_arithmetic(bill), "summary": charge_summary(bill)}


@app.post("/split")
def split(payload: dict):
    bill = clean_bill(payload.get("bill"))
    if not bill["items"]:
        raise HTTPException(400, "The bill has no items.")
    people = clean_people(payload.get("people"))
    try:
        asg = {int(k): {str(n): int(u) for n, u in v.items()} for k, v in (payload.get("assignments") or {}).items()}
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, "assignments must map item numbers to {name: share units}")
    try:
        res = split_bill(bill, people, asg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ov = payload.get("overrides") or {}
    if ov:
        unknown = [k for k in ov if k not in people]
        if unknown:
            raise HTTPException(400, f"Unknown person in overrides: {unknown}")
        res = apply_overrides(res, {k: to_paise(v) for k, v in ov.items()}, bool(payload.get("rebalance")))
    return {"result": res, "people": people, "summary": charge_summary(bill), "explanation": explain_all(bill, res)}


@app.post("/export")                                        # the problem statement's JSON (numbers, readable charge names)
def export(payload: dict):
    return to_ps_schema(clean_bill(payload.get("bill")))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


EXAMPLES = Path(__file__).parent.parent / "examples" / "demo_bills"


@app.get("/examples")
def examples():
    """Photos in `examples/demo_bills/`, offered on the first screen (they go through the real model like any upload)."""
    if not EXAMPLES.exists():
        return {"files": []}
    return {"files": sorted(f.name for f in EXAMPLES.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))}


if EXAMPLES.exists():
    app.mount("/example", StaticFiles(directory=EXAMPLES), name="example")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
