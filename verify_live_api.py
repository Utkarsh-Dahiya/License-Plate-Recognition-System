"""
LIVE API VERIFICATION (diagnostic, not app runtime).

Starts against an already-running uvicorn instance and checks the things
the app must guarantee in production:

  1. GET /api/health    -> schema unchanged
  2. GET /api/readiness -> reports model state (additive endpoint)
  3. POST /api/detect/image -> real response schema on a real val image
  4. /api/health STAYS RESPONSIVE while a heavy detection is in flight
     (the event loop must not be blocked by CPU-bound inference)
  5. multi-plate frame -> every plate gets its own box + OCR entry
  6. two-line (motorcycle) plate -> band split produces a joined read

The script SPAWNS its own uvicorn subprocess (so the checks always run
against a single-process server exercising the real lifespan warmup),
waits for it to answer /api/health, runs every check, then tears the
server down. Nothing is left listening on the port afterwards.

Usage:
    venv/Scripts/python.exe verify_live_api.py
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app" / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

# Exactly the three keys /api/health has always returned. The requirement
# is that this schema does NOT change; a fourth key here is a regression.
HEALTH_KEYS = {"status", "models_loaded", "sources"}


def port_is_open(host, port) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_health(base, timeout_s=90) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{base}/api/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def post_image(base, path, conf=None, timeout=180):
    url = f"{base}/api/detect/image"
    if conf is not None:
        url += f"?conf={conf}"
    with open(path, "rb") as fh:
        return requests.post(url, files={"file": (Path(path).name, fh, "image/jpeg")},
                             timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--no-server", action="store_true",
                    help="use an already-running server on --port")
    args = ap.parse_args()

    host = "127.0.0.1"
    base = f"http://{host}:{args.port}"

    server = None

    if not args.no_server:
        if port_is_open(host, args.port):
            print(f"port {args.port} already in use — stop that process first.")
            sys.exit(2)

        print(f"starting uvicorn on {base} ...")
        server = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "main:app",
                "--host", host, "--port", str(args.port),
                "--log-level", "warning",
            ],
            cwd=str(ROOT / "app" / "backend"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "LVA_DEBUG_TIMING": "0"},
        )

        # Guarantee teardown even if a check raises mid-run.
        import atexit

        atexit.register(server.terminate)

        if not wait_for_health(base, timeout_s=120):
            print("server did not become healthy in time")
            server.terminate()
            sys.exit(2)
        print("server healthy")

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # ---------------- 1. health ----------------
    print("\n[1] GET /api/health")
    h = requests.get(f"{base}/api/health", timeout=20)
    body = h.json()
    check("HTTP 200", h.status_code == 200, f"status={h.status_code}")
    check("schema unchanged", set(body.keys()) == HEALTH_KEYS,
          f"keys={sorted(body.keys())}")

    # ---------------- 2. readiness ----------------
    print("\n[2] GET /api/readiness")
    r = requests.get(f"{base}/api/readiness", timeout=20)
    rb = r.json()
    check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
    check("reports model state", "models_loaded" in rb,
          f"models_loaded={rb.get('models_loaded')} status={rb.get('status')}")

    # ---------------- 3. detection schema ----------------
    val = ROOT / "YOLO_dataset" / "images" / "val"
    imgs = sorted(val.glob("*.png")) + sorted(val.glob("*.jpeg"))

    # Warm up first (model load may still be running).
    warm_path = str(imgs[0])
    t0 = time.perf_counter()
    w = post_image(base, warm_path)
    print(f"  (warmup detect: {time.perf_counter() - t0:.1f}s, "
          f"HTTP {w.status_code})")
    check("warmup detect succeeded", w.status_code == 200, f"HTTP {w.status_code}")

    print("\n[3] POST /api/detect/image -> response schema")
    resp = post_image(base, warm_path)
    d = resp.json()
    top = {"image", "plates_detected", "plates", "annotated_image_base64",
           "processing_time_seconds"}
    check("top-level keys unchanged", set(d.keys()) == top,
          f"keys={sorted(d.keys())}")
    check("image width/height present",
          isinstance(d.get("image", {}).get("width"), int)
          and isinstance(d.get("image", {}).get("height"), int),
          str(d.get("image")))
    plate_keys = {"plate_id", "bbox", "yolo_confidence", "ocr_text",
                  "ocr_confidence", "validation_score", "validation",
                  "final_confidence", "status"}
    if d["plates"]:
        p = d["plates"][0]
        check("plate keys present", plate_keys <= set(p.keys()),
              f"missing={sorted(plate_keys - set(p.keys()))}")
        b = p["bbox"]
        check("bbox is 4 ints in image space",
              len(b) == 4 and all(isinstance(v, int) for v in b)
              and 0 <= b[0] < b[2] <= d["image"]["width"]
              and 0 <= b[1] < b[3] <= d["image"]["height"],
              f"bbox={b} image={d['image']}")
        check("no debug payload leaked", "debug" not in d
              and "debug_ocr_crop_b64" not in p)
    else:
        check("at least one plate detected on warm image", False)

    # ---------------- 4. health responsive during detection ----------------
    print("\n[4] /api/health WHILE detection is running (event loop not blocked)")
    # Pick the largest available image so the job actually takes time.
    biggest = max(imgs, key=lambda p: p.stat().st_size)
    latencies = []
    done = threading.Event()

    def heavy():
        try:
            post_image(base, str(biggest))
        finally:
            done.set()

    t = threading.Thread(target=heavy)
    t.start()
    time.sleep(0.35)  # let the detection actually be in flight
    while not done.is_set() and len(latencies) < 40:
        t0 = time.perf_counter()
        try:
            requests.get(f"{base}/api/health", timeout=10).json()
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            latencies.append(float("inf"))
        time.sleep(0.05)
    t.join()

    if latencies:
        med = statistics.median(latencies)
        worst = max(latencies)
        print(f"    {len(latencies)} health polls during detection: "
              f"median={med:.0f} ms max={worst:.0f} ms")
        check("health median < 500 ms under load", med < 500, f"{med:.0f} ms")
        check("no health request timed out under load",
              all(v != float("inf") for v in latencies))
    else:
        check("health polled during detection", False)

    # ---------------- 5. multi-plate ----------------
    print("\n[5] multi-plate frame")
    collage = ROOT / "YOLO_dataset" / "images" / "val" / "Cars266.png"
    if collage.exists():
        md = post_image(base, str(collage)).json()
        ids = [p["plate_id"] for p in md["plates"]]
        boxes = [tuple(p["bbox"]) for p in md["plates"]]
        check("more than one plate reported", len(md["plates"]) > 1,
              f"plates={len(md['plates'])}")
        check("plate_ids unique", len(set(ids)) == len(ids), str(ids))
        check("boxes are distinct (no duplicate box)",
              len(set(boxes)) == len(boxes), str(boxes))
        for p in md["plates"]:
            print(f"      #{p['plate_id']} bbox={p['bbox']} "
                  f"yolo={p['yolo_confidence']} text={p['ocr_text']!r} "
                  f"final={p['final_confidence']}")
    else:
        print("      (Cars266.png not found, skipped)")

    # ---------------- 6. two-line plate ----------------
    print("\n[6] two-line (motorcycle) plate")
    sys.path.insert(0, str(ROOT))
    from diagnose_plates import make_two_plate_scene  # noqa: E402

    scene = make_two_plate_scene([("37-N1", "4635")])
    ok, buf = cv2.imencode(".jpg", scene)
    tmp = ROOT / "batch_results" / "ocr_benchmark" / "_twoline_probe.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(buf.tobytes())
    try:
        td = post_image(base, str(tmp)).json()
        print(f"      plates_detected={td['plates_detected']}")
        for p in td["plates"]:
            print(f"      bbox={p['bbox']} yolo={p['yolo_confidence']} "
                  f"text={p['ocr_text']!r} conf={p['ocr_confidence']} "
                  f"status={p['status']}")
        got_text = any(p["ocr_text"] for p in td["plates"])
        check("two-line plate produced a non-empty read", got_text,
              "no text extracted")
        # The bands are '37-N1' and '4635'; a correct split+join contains
        # the digits of both.
        joined = "".join(p["ocr_text"] or "" for p in td["plates"])
        check("read contains both band digit groups",
              ("37" in joined and "4635" in joined),
              f"joined={joined!r}")
    finally:
        tmp.unlink(missing_ok=True)

    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILURES ({len(failures)}): {failures}")
        sys.exit(1)
    print("ALL LIVE API CHECKS PASSED")


if __name__ == "__main__":
    main()
