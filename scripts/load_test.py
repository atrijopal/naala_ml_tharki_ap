"""Concurrent-load test of a running app (/extract with the real model). Every upload gets a few random trailing bytes so the result cache cannot answer it.
While the model is busy, /validate is probed to see whether the rest of the server stays responsive.
  python scripts/load_test.py [--url http://localhost:8765] [--levels 1,2,4,8] [--per_client 5] [--out results/load_test.json]"""
import argparse, glob, json, os, statistics, threading, time, urllib.request, uuid

ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8765"); ap.add_argument("--levels", default="1,2,4,8")
ap.add_argument("--per_client", type=int, default=5); ap.add_argument("--out", default="results/load_test.json"); ap.add_argument("--images", default="examples/demo_bills/*.jpg"); a = ap.parse_args()
imgs = [open(f, "rb").read() for f in sorted(glob.glob(a.images))]


def extract(data):
    b = uuid.uuid4().hex; data = data + os.urandom(8)
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="b.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
    t = time.perf_counter(); urllib.request.urlopen(urllib.request.Request(a.url + "/extract", body, {"Content-Type": f"multipart/form-data; boundary={b}"}), timeout=300).read()
    return time.perf_counter() - t


def probe(stop, out):
    bill = json.dumps({"bill": {"items": [{"name": "x", "qty": "1", "unit_price": "10", "total": "10"}], "charges": [], "subtotal": "10", "grand_total": "10"}}).encode()
    while not stop.is_set():
        t = time.perf_counter()
        try: urllib.request.urlopen(urllib.request.Request(a.url + "/validate", bill, {"Content-Type": "application/json"}), timeout=60).read(); out.append(time.perf_counter() - t)
        except Exception: out.append(float("inf"))
        time.sleep(0.15)


extract(imgs[0])                                              # warm
res = []
for n in [int(x) for x in a.levels.split(",")]:
    lat, errs, pr, stop = [], [], [], threading.Event()
    def client(k):
        for j in range(a.per_client):
            try: lat.append(extract(imgs[(k * a.per_client + j) % len(imgs)]))
            except Exception as e: errs.append(str(e)[:80])
    pt = threading.Thread(target=probe, args=(stop, pr)); pt.start()
    ths = [threading.Thread(target=client, args=(k,)) for k in range(n)]; t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]; wall = time.perf_counter() - t0; stop.set(); pt.join()
    lat.sort(); r = {"clients": n, "requests": len(lat), "errors": len(errs), "wall_s": round(wall, 2), "bills_per_s": round(len(lat) / wall, 2),
                     "latency_p50_s": round(lat[len(lat) // 2], 2), "latency_p95_s": round(lat[min(int(len(lat) * 0.95), len(lat) - 1)], 2), "latency_max_s": round(lat[-1], 2),
                     "validate_probe_max_ms": round(1000 * max(pr), 1) if pr else None, "validate_probe_median_ms": round(1000 * statistics.median(pr), 1) if pr else None}
    res.append(r); print("LOAD", json.dumps(r), flush=True)
json.dump({"url": a.url, "levels": res}, open(a.out, "w"), indent=1)
