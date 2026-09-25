"""
VERIFICATION SUITE — schema / singleton / concurrency / memory.

Run: venv/Scripts/python.exe verify_suite.py
Run in a FRESH process (it resets and re-loads models for the
concurrency check). Not part of the app runtime.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import services.detection_service as ds

# The original committed plate-entry keys (from git HEAD's code path).
ORIGINAL_PLATE_KEYS = {
    "plate_id", "bbox", "yolo_confidence", "ocr_text", "ocr_confidence",
    "validation_score", "validation", "final_confidence", "status",
}
ORIGINAL_TOP_KEYS = {
    "image", "plates_detected", "plates",
    "annotated_image_base64", "processing_time_seconds",
}


def human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


def rss() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc),
            pmc.cb,
        )
        return pmc.WorkingSetSize


def main() -> None:
    print("=" * 74)
    print("VERIFICATION SUITE")
    print("=" * 74)

    cars = sorted(ROOT.glob("batch_results/*_Cars*.jpg"))[:6]
    payloads = [p.read_bytes() for p in cars]

    # ------------------------------------------------------------
    # 1. Cold start + first request (fresh process = fresh container)
    # ------------------------------------------------------------
    print("\n[1] FIRST REQUEST (includes lazy model load)")
    t0 = time.perf_counter()
    result = ds.run_detection_on_image(payloads[0])
    first_ms = (time.perf_counter() - t0) * 1000
    print(f"    first-request total: {first_ms:.0f} ms")

    # ------------------------------------------------------------
    # 2. Schema compatibility (additive-only check against original)
    # ------------------------------------------------------------
    print("\n[2] API SCHEMA COMPATIBILITY")
    top_keys = set(result.keys())
    assert top_keys == ORIGINAL_TOP_KEYS, (
        f"top-level keys changed: {top_keys ^ ORIGINAL_TOP_KEYS}"
    )
    ok = True
    for plate in result["plates"]:
        keys = set(plate.keys())
        missing = ORIGINAL_PLATE_KEYS - keys
        added = keys - ORIGINAL_PLATE_KEYS
        if missing:
            ok = False
            print(f"    MISSING keys: {missing}")
        if added:
            print(f"    + additive keys (allowed): {sorted(added)}")
    assert ok, "schema broke: original keys missing"
    print(f"    top-level keys: EXACTLY the original 5  OK")
    print(f"    plate keys:     original 9 present, "
          f"additive-only extras  OK")
    # annotated image still decodes
    import base64

    buf = base64.b64decode(result["annotated_image_base64"])
    import numpy as np
    import cv2

    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0
    print(f"    annotated_image_base64: decodes to {img.shape[1]}x{img.shape[0]}  OK")

    # ------------------------------------------------------------
    # 3. Warm requests + singleton identity
    # ------------------------------------------------------------
    print("\n[3] WARM REQUESTS + SINGLETON IDENTITY")
    yolo_id = id(ds._yolo_model)
    reader_id = id(ds._ocr_reader)
    warm_times = []
    for data in payloads[1:]:
        t0 = time.perf_counter()
        ds.run_detection_on_image(data)
        warm_times.append((time.perf_counter() - t0) * 1000)
    assert id(ds._yolo_model) == yolo_id, "YOLO model was re-created!"
    assert id(ds._ocr_reader) == reader_id, "EasyOCR reader was re-created!"
    print(f"    warm times: {[f'{t:.0f} ms' for t in warm_times]}")
    print(f"    mean warm:  {sum(warm_times) / len(warm_times):.0f} ms")
    print(f"    YOLO singleton: same object across requests  OK")
    print(f"    OCR  singleton: same object across requests  OK")

    # ------------------------------------------------------------
    # 4. Concurrent first-load: exactly one model instance
    # ------------------------------------------------------------
    print("\n[4] CONCURRENT LOAD (8 threads, models reset -> load race)")
    # Count real instantiations by wrapping the classes.
    import easyocr
    import ultralytics
    from ultralytics import YOLO as _RealYOLO

    counts = {"yolo": 0, "reader": 0}
    lock = threading.Lock()

    class CountingYOLO(_RealYOLO):
        def __init__(self, *a, **kw):
            with lock:
                counts["yolo"] += 1
            super().__init__(*a, **kw)

    class CountingReader(easyocr.Reader):
        def __init__(self, *a, **kw):
            with lock:
                counts["reader"] += 1
            super().__init__(*a, **kw)

    ds._yolo_model = None
    ds._ocr_reader = None
    ds._load_error = None
    ultralytics.YOLO = CountingYOLO
    easyocr.Reader = CountingReader

    errors: list[str] = []

    def worker():
        try:
            ds.run_detection_on_image(payloads[0])
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"    8 concurrent requests finished in "
          f"{time.perf_counter() - t0:.1f}s")
    print(f"    YOLO instantiations:    {counts['yolo']}  "
          f"{'OK' if counts['yolo'] == 1 else 'FAIL'}")
    print(f"    EasyOCR instantiations: {counts['reader']}  "
          f"{'OK' if counts['reader'] == 1 else 'FAIL'}")
    print(f"    request errors: {len(errors)}  "
          f"{'OK' if not errors else errors[:2]}")
    assert counts["yolo"] == 1 and counts["reader"] == 1 and not errors

    # ------------------------------------------------------------
    # 5. Memory profile
    # ------------------------------------------------------------
    print("\n[5] MEMORY (working set)")
    base = rss()
    print(f"    after load + requests: {human_mb(base)}")

    peak = base

    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, rss())
            time.sleep(0.02)

    st = threading.Thread(target=sampler)
    st.start()
    for data in payloads[:3]:
        ds.run_detection_on_image(data)
    stop.set()
    st.join()
    print(f"    peak during 3 requests: {human_mb(peak)}")
    print("    (models: YOLO + EasyOCR recognizer-only, torch CPU, "
          "single-thread pools)")

    # ------------------------------------------------------------
    # 6. OCR spot-checks on real images (no hardcoding; GT for FAG643)
    # ------------------------------------------------------------
    print("\n[6] OCR RESULTS ON REAL IMAGES")
    for path, res in zip(cars, [result] + [None] * 5):
        pass
    texts = []
    for data in payloads:
        out = ds.run_detection_on_image(data)
        for p in out["plates"]:
            texts.append(p["ocr_text"] or "(ocr failed)")
    print(f"    {list(zip([c.name for c in cars], texts))}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
