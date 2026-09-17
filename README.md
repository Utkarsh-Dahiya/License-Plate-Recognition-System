# License Vision AI

Automatic license plate recognition (ALPR) with a FastAPI + React console on top of a trained YOLO detector and EasyOCR.

The UI **serves existing batch and video results**. It does not retrain YOLO or re-run the 4K video when you open the dashboard.

## What is real

Figures below are copied from files already in this repo. They are **not** character-level OCR accuracy.

| Source | Metric | Value | Meaning |
|---|---|---|---|
| `batch_results/analytics/dashboard_data.json` | Images | 658 | Batch run size |
| same | Plates detected | 765 | YOLO boxes across those images |
| same | OCR success rate | 93.31% | Share of images with status `SUCCESS` (EasyOCR returned **some** text, not necessarily a correct plate) |
| same | Mean YOLO confidence | 80.16% | From `batch_results.csv` |
| same | Mean OCR confidence | 68.92% | From `batch_results.csv` |
| `video_results/video_summary.json` | Sampled frames | 720 | 3600-frame 4K clip, skip 5, processed at 1280px wide |
| same | Detections | 1013 | Frame-level boxes with OCR attempts |
| same | Unique plates | 650 | Unique **OCR strings** (near-duplicates are counted separately) |
| `runs/detect/models/license_plate_detector/results.csv` (epoch 30) | Precision | 0.92725 | YOLO **detection** validation, not OCR |
| same | Recall | 0.86897 | same |
| same | mAP50 | 0.91216 | same |
| same | mAP50-95 | 0.54929 | same |

## Layout

```
app/backend/          FastAPI API (run this)
app/frontend/         React + Vite + Tailwind UI
batch_results/        Saved 658-image CSV + dashboard JSON
video_results/        Saved video CSV/JSON (+ local annotated mp4)
runs/detect/models/   Training `results.csv`, `args.yaml`, local `best.pt`
yolo_dataset/         Labels + `data.yaml` (images are local / gitignored)
plate_pipeline.py     Offline EasyOCR image pipeline
batch_test.py         Offline batch over dataset images
video_pipeline.py     Offline video engine (writes *consensus* filenames)
alpr.py               Older Tesseract experiment (not used by the app)
license-vision-ai/    Leftover earlier drop of the app — **do not run this**; use `app/`
```

## Run the product

Weights must exist locally at:

`runs/detect/models/license_plate_detector/weights/best.pt`

They are gitignored (`*.pt`). Copy your trained `best.pt` there before live image detection.

**Backend** (from `app/backend`, Python 3.11+ recommended):

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The repo may already have a root `venv/` from earlier work; you can reuse that if it has the requirements installed.

**Frontend** (from `app/frontend`):

```powershell
npm install
npm run dev
```

Open `http://localhost:5173`. API default: `http://localhost:8000` (`VITE_API_BASE` to override).

Check `http://localhost:8000/api/health` — source files should show `"exists": true` for CSV/JSON (and weights/video if those local files are present).

### Environment variables (optional)

| Variable | Purpose |
|---|---|
| `LVA_PROJECT_ROOT` | Project root if the backend is not at `app/backend` |
| `LVA_MODEL_PATH` | Path to `best.pt` |
| `LVA_BATCH_CSV` / `LVA_DASHBOARD_JSON` | Batch result files |
| `LVA_VIDEO_DETECTIONS_CSV` / `LVA_VIDEO_SUMMARY_JSON` / `LVA_ANNOTATED_VIDEO` | Video result files |
| `VITE_API_BASE` | Frontend API origin |

## Live vs saved pipelines

- **Overview / Batch / Video stats / Registry** read saved CSV/JSON. Opening those pages does not run YOLO.
- **Live / Image Analysis** runs YOLO + EasyOCR on the upload (`app/backend/services/detection_service.py`).
- **Process a new video** is an explicit demo job. It does **not** overwrite `video_results/` analytics from the 4K run.

Offline script details (actual hardcoded paths, no CLI flags) are in `README.txt`.

## License / privacy

This is a local demo: no authentication. Do not expose the API to the public internet without access control. Training images and 4K video are not intended for a public git clone.
