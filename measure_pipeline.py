"""
PIPELINE MEASUREMENT — where do the seconds and megabytes go?

Measures, in one fresh process (mirrors a cold Render container):
  1. import times (torch / ultralytics / easyocr)
  2. YOLO load + warmup
  3. EasyOCR Reader load
  4. per-stage request timing (YOLO / crop / preprocess / OCR / scoring)
  5. RSS at every stage boundary + peak
  6. OCR pass counts per plate
  7. YOLO conf-threshold sweep (box count + conf) on real images

Run: LVA_DEBUG_TIMING=0 venv/Scripts/python.exe measure_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app" / "backend"))


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
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
        return pmc.WorkingSetSize / (1024 * 1024)


def peak_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
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
        return pmc.PeakWorkingSetSize / (1024 * 1024)


def main() -> None:
    print("=" * 74)
    print("COLD-START DECOMPOSITION (fresh process = fresh container)")
    print("=" * 74)

    t = time.perf_counter()
    m0 = rss_mb()
    import numpy as np  # noqa: F401

    t_np = time.perf_counter() - t

    t = time.perf_counter()
    import cv2  # noqa: F401

    t_cv = time.perf_counter() - t

    t = time.perf_counter()
    import torch

    t_torch = time.perf_counter() - t
    m_torch = rss_mb()

    t = time.perf_counter()
    import ultralytics
    from ultralytics import YOLO  # noqa: F401

    t_ultra = time.perf_counter() - t
    m_ultra = rss_mb()

    t = time.perf_counter()
    import easyocr

    t_easy = time.perf_counter() - t
    m_easy = rss_mb()

    print(f"  numpy        {t_np*1000:7.0f} ms")
    print(f"  cv2          {t_cv*1000:7.0f} ms")
    print(f"  torch        {t_torch*1000:7.0f} ms   rss {m_torch:6.0f} MB")
    print(f"  ultralytics  {t_ultra*1000:7.0f} ms   rss {m_ultra:6.0f} MB")
    print(f"  easyocr      {t_easy*1000:7.0f} ms   rss {m_easy:6.0f} MB")

    import services.detection_service as ds

    weights = ds.MODEL_WEIGHTS_PATH

    t = time.perf_counter()
    yolo = YOLO(str(weights))
    t_yload = time.perf_counter() - t
    m_yload = rss_mb()

    t = time.perf_counter()
    with torch.inference_mode():
        yolo.predict(
            source=np.zeros((320, 320, 3), dtype=np.uint8),
            imgsz=640,
            conf=0.25,
            verbose=False,
        )
    t_ywarm = time.perf_counter() - t
    m_ywarm = rss_mb()

    t = time.perf_counter()
    reader = easyocr.Reader(["en"], gpu=False, detector=False, verbose=False)
    t_eload = time.perf_counter() - t
    m_eload = rss_mb()

    print(f"  YOLO load    {t_yload*1000:7.0f} ms   rss {m_yload:6.0f} MB")
    print(f"  YOLO warmup  {t_ywarm*1000:7.0f} ms   rss {m_ywarm:6.0f} MB")
    print(f"  EasyOCR load {t_eload*1000:7.0f} ms   rss {m_eload:6.0f} MB")
    print(f"  TOTAL COLD   {(time.perf_counter()-t_np):7.1f} s   rss {m_eload:6.0f} MB")

    ds._yolo_model = yolo
    ds._ocr_reader = reader
    ds._low_memory_caps_applied = True

    # ------------------------------------------------------------
    # Per-stage warm timing + OCR pass counts
    # ------------------------------------------------------------
    cars = sorted(ROOT.glob("batch_results/*_Cars*.jpg"))[:8]
    payloads = [p.read_bytes() for p in cars]

    print("\n" + "=" * 74)
    print("WARM PER-STAGE TIMING (via service timing buckets)")
    print("=" * 74)

    ds.logger.setLevel(ds.logging.WARNING)

    import io
    from contextlib import redirect_stderr

    for name, data in zip([p.name for p in cars], payloads):
        buf_out = io.StringIO()
        with redirect_stderr(buf_out):
            t0 = time.perf_counter()
            out = ds.run_detection_on_image(data)
            total = (time.perf_counter() - t0) * 1000
        print(
            f"  {name:<18} total={total:6.0f} ms  "
            f"plates={out['plates_detected']}  "
            f"texts={[p['ocr_text'] for p in out['plates']]}"
        )

    print(f"  peak RSS so far: {peak_mb():.0f} MB")

    # ------------------------------------------------------------
    # YOLO conf sweep: how does threshold affect box count / quality?
    # ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("YOLO CONF-THRESHOLD SWEEP (real images, warm)")
    print("=" * 74)

    for data in payloads[:4]:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            s = 1280 / max(h, w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        row = [Path(data).stem if isinstance(data, Path) else "?"]
        print(f"\n  image bytes={len(data)//1024} KB decoded={img.shape[1]}x{img.shape[0]}")
        for conf in (0.10, 0.25, 0.40, 0.55):
            with torch.inference_mode():
                res = yolo.predict(source=img, imgsz=640, conf=conf, verbose=False)
            boxes = res[0].boxes
            n = 0 if boxes is None else len(boxes)
            confs = (
                [round(float(c), 2) for c in boxes.conf.tolist()]
                if n
                else []
            )
            print(f"    conf>={conf:.2f}: boxes={n}  confs={confs}")

    print(f"\nFINAL peak RSS: {peak_mb():.0f} MB")


if __name__ == "__main__":
    main()
