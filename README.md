# License Plate Detection & OCR System

A license-plate detection and OCR platform: a trained **YOLO** detector plus an
**EasyOCR** pipeline behind a **FastAPI** API and a **React + Vite + Tailwind**
console. It localises vehicle plates, reads the characters, validates the string
against real Indian registration formats, resolves state codes, and keeps a
searchable run history and plate registry.

The dashboard **serves existing batch and video results** and runs **live
detection** on uploads. Opening a page never retrains the model, and it never
re-runs the offline 4K video.

---

## Overview

- **What it is** — an end-to-end Automatic License Plate Recognition (ALPR)
  application built around an already-trained detector and an OCR cascade.
- **What it does** — detect plates in a still image, read and validate the plate
  string, and present detection/adoption analytics from saved batch and video runs.
- **What it is not** — it does not train models in the browser, does not claim
  character-level OCR accuracy, and does not fabricate metrics. Missing data is
  reported as "N/A".

## Core capabilities

- Live image detection (YOLO localisation + adaptive EasyOCR cascade).
- Indian registration validation (current 2005+ series and legacy series).
- State-code → state-name resolution.
- Two-line (stacked / motorcycle) plate handling.
- CSV / JSON export of detection results and of the run history.
- Batch analytics over an existing 658-image run.
- Video analytics over an existing 4K run, plus a background "process a new
  video" demo job.
- Plate registry built from the saved video run.
- Model performance view backed by real training `results.csv`.
- Append-only run history with search, pagination, statistics and export.
- System page showing engine status and which data files were found.

## Architecture

```
        Browser (React + Vite + Tailwind)
                     │  fetch (VITE_API_BASE)
                     ▼
        FastAPI  (app/backend/main.py)  ── 21 application endpoints
          ├─ services/data_loader.py        reads saved CSV/JSON (mtime-cached)
          ├─ services/detection_service.py  YOLO + EasyOCR (singleton, lazy)
          └─ services/history_log.py        append-only JSONL run log
                     │
     ┌───────────────┴────────────────┐
     ▼                                ▼
  saved results (CSV/JSON)      live inference (on upload)
  batch_results/, video_results/    YOLO detector + EasyOCR
```

Model-heavy work (YOLO + OCR, and one video job) runs on a **dedicated bounded
thread pool** so light endpoints (health, dashboard, status polling) stay
responsive. Models load **once** behind a lock and are **warmed up in the
background** at startup; `/api/readiness` reports the warm-up state.

## YOLO detection pipeline

- Single-class license-plate detector (Ultralytics YOLO).
- **Two-pass policy:** a primary pass at 640 px (the training size); a retry pass
  at higher resolution only when the primary pass found nothing plate-like or its
  best box is weak.
- Cross-pass **IoU de-duplication** (primary boxes are preferred on merge).
- Geometric **plausibility checks** (aspect ratio / size sanity) used to report
  which detections look plate-shaped — they never delete boxes.
- Conservative box expansion before cropping so characters at the edges are kept.
- Full-image inference is capped at a maximum side length for predictable memory.

## OCR pipeline

EasyOCR runs in **recogniser-only** mode (the detector has already isolated the
plate), driven by an evidence-gated cascade that stops as soon as the result is
well supported:

1. **Tier 1 — CLAHE** enhancement (primary path).
2. **Tier 4 — two-band split** for tall/stacked plates (runs early so a stacked
   plate is not mis-merged into one line).
3. **Tier 2 — OTSU** threshold fallback.
4. **Tier 3 — denoise + OTSU** last resort.

Additional behaviour:

- A width-preserving, bounded OCR canvas (no aspect distortion).
- Skew estimation that only de-skews when the geometry provides clear evidence.
- Low-confidence boxes get a bounded effort (one cheap pass) instead of the full
  cascade; boxes whose whole cascade is empty get one wider-context retry.
- Per-character CTC probabilities are captured (behaviour-preserving) to score the
  leading/trailing characters of every candidate.

## Indian license plate validation

Candidates are validated against the **real** Indian registration format:

- **Current series** — `SS` state code + two-digit RTO (01-99, leading zero
  required; well past 30 for big states) + optional series letters + 1-4 digits.
- **Legacy series** — older all-letter prefixes still on the road.
- State codes must be **genuine** state/UT codes; district numbers are sanity-checked.
- Delhi heritage single-digit and letter-suffixed RTOs (e.g. `DL 3C`) are supported.

Validation produces a score and a classification (`INDIAN_PLATE`,
`POSSIBLE_INDIAN_PLATE`, `UNVERIFIED`).

## State-code / state-name resolution

The first two characters are matched against a full state/UT code table and the
human-readable state name is returned alongside the detection, so the UI can show
both the code and the state.

## Two-line plate handling

Motorcycle-style plates stack the state code over the serial. Feeding such a crop
to the recogniser as one line produces garbage, so the cascade splits the crop
into horizontal text bands, recognises each band with the same reader, and joins
the reads top-to-bottom — while preserving the band evidence for scoring. The
whole-crop canvas is prevented from collapsing the stacked plate back to one line.


## Measured results

Every figure below is copied from files already committed to this repository —
none is invented. These are detection and **text-extraction** metrics; they are
**not** character-level OCR accuracy.

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

## Image detection

`POST /api/detect/image` accepts a multipart image upload (max 10 MB) and returns
the detection result. It is the only endpoint that runs live inference; the
recogniser path is the same cascade described above.

The response preserves a stable, backward-compatible schema:

| Key | Meaning |
|---|---|
| `image` | `{ width, height }` of the processed image |
| `plates_detected` | number of plates returned |
| `plates` | list of per-plate objects |
| `annotated_image_base64` | JPEG (base64) with boxes and labels drawn |
| `processing_time_seconds` | server-side wall-clock time |

Each entry in `plates` includes `plate_id`, `bbox`, `yolo_confidence`, `ocr_text`,
`ocr_confidence`, `validation_score`, `validation`, `final_confidence`, `status`,
and additive fields `state_code`, `state_name`, `strict_format`, `edge_confidence`.
`status` is one of `HIGH_CONFIDENCE` (≥ 0.80), `REVIEW` (≥ 0.50),
`LOW_CONFIDENCE`, or `OCR_FAILED`. An optional `conf` query parameter overrides
the detector threshold; when omitted, the tuned server default applies.

## Batch processing

An offline batch pipeline already processed a 658-image dataset and wrote
`batch_results/batch_results.csv` plus `batch_results/analytics/dashboard_data.json`.
The API serves that data (it does not recompute it on load):

- `GET /api/batch/results` — paginated, filterable rows (`status`, `search`, `page`).
- `GET /api/batch/analytics` — aggregate totals and rates.

## Video analytics

An offline video pipeline processed a 4K clip and wrote
`video_results/detections.csv` and `video_results/video_summary.json`. The API
serves those files, and the app can also stream the locally produced annotated
video:

- `GET /api/video/summary` — clip metadata + detection/OCR totals.
- `GET /api/video/detections` — paginated frame-level observations.
- `POST /api/process/video` + `GET /api/process/video/{job_id}` — an explicit,
  separate **demo job** that samples frames of a small upload. It does **not**
  overwrite the saved 4K analytics, and it never re-runs the 4K source.

## Plate registry & history

- **Plate registry** — built from the saved video run's plate list.
  `GET /api/plates` (search / sort) and `GET /api/plates/{plate_text}` (detail).
- **Run history** — every real `/api/detect/image` and `/api/process/video` call is
  appended to a JSONL log (`app/backend/data/history/detections.jsonl`):
  - `GET /api/history` — paginated, searchable, type-filtered.
  - `GET /api/history/stats` — totals, text rate, mean confidence, latency, status mix.
  - `GET /api/history/export` — CSV or JSON download of the (filtered) history.

## Analytics & diagnostics

- `GET /api/dashboard` — the Overview KPIs and charts.
- `GET /api/model` — detector/OCR description, training metrics, observed performance.
- `GET /api/system` — stack info and which data files were found (with paths).
- `GET /api/samples`, `/api/samples/{id}`, `/api/samples/{id}/image` — curated
  one-click test images.
- `POST /api/debug/plate-crop` — debug-only; returns the exact OCR input crop.
  Disabled unless `LVA_DEBUG_MODE=1` **and** the caller opts in.

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + which sources exist |
| GET | `/api/readiness` | Model warm-up state (`ready` / `warming_up` / `error`) |
| GET | `/api/dashboard` | Overview KPIs and charts |
| GET | `/api/batch/results` | Paginated/filtered batch rows |
| GET | `/api/batch/analytics` | Batch aggregates |
| GET | `/api/video/summary` | Saved video summary |
| GET | `/api/video/detections` | Saved frame-level detections |
| GET | `/api/plates` | Plate registry list |
| GET | `/api/plates/{plate_text}` | Plate registry detail |
| GET | `/api/model` | Model + training information |
| GET | `/api/system` | Runtime + data-source status |
| GET | `/api/history` | Run history (paginated/searchable) |
| GET | `/api/history/stats` | Run-history statistics |
| GET | `/api/history/export` | Export history (CSV/JSON) |
| GET | `/api/samples` | Curated sample list |
| GET | `/api/samples/{id}` | Sample metadata |
| GET | `/api/samples/{id}/image` | Sample image bytes |
| POST | `/api/detect/image` | Live detection on an upload |
| POST | `/api/process/video` | Queue a demo video job |
| GET | `/api/process/video/{job_id}` | Poll a demo video job |
| POST | `/api/debug/plate-crop` | Debug crop (opt-in) |

Interactive docs are available at `/docs` when the server is running.

## Frontend overview

A React 18 + Vite + Tailwind single-page console with a dark, instrument-panel
style. Routes:

| Route | Page |
|---|---|
| `/` | Overview — KPIs, charts, recent runs, top plates |
| `/detection` | Detection history — filters, search, statistics, export |
| `/image-analysis` | Live studio — upload or pick a sample, run detection, inspect results |
| `/batch` | Batch Intelligence — search/filter the saved batch |
| `/video` | Video Analytics — saved stats, annotated player, demo job |
| `/registry` | Plate Registry — searchable list with a detail drawer |
| `/model` | Model Performance — detector/OCR info and real training metrics |
| `/system` | System — engines and data-source status |

`src/lib/api.js` is the single fetch wrapper: it applies per-call timeouts, surfaces
typed errors, and refuses to silently call `localhost` in a production build (it
requires `VITE_API_BASE`). Every page guards its data calls, so an API failure
shows an empty/error state instead of crashing the UI.

## Project structure

```
app/backend/                    FastAPI API (run this)
  main.py                       All endpoints, CORS, warm-up, thread pools
  requirements.txt              Backend runtime dependencies
  .env.example                  Backend environment template
  services/
    data_loader.py              Reads saved CSV/JSON (mtime-cached)
    detection_service.py        YOLO + EasyOCR detection/OCR core
    history_log.py              Append-only run history + stats/export
  data/history/detections.jsonl Real run log (append-only)
app/frontend/                   React + Vite + Tailwind UI
  src/pages/                    One file per route
  src/components/               Sidebar, panels, cards, viewport
  src/hooks/useDetection.js     Detection state machine
  src/lib/api.js                Typed fetch wrapper
batch_results/                  Saved 658-image CSV + dashboard JSON
video_results/                  Saved video CSV/JSON (+ local annotated mp4)
runs/detect/models/             Training results.csv, args.yaml, local best.pt
YOLO_dataset/, test_images/     Labeled data + data.yaml (images local/gitignored)
tests/                          Unit tests (unittest)
plate_pipeline.py, video_pipeline.py, batch_test.py   Offline scripts
verify_suite.py, verify_live_api.py                    Verification harnesses
requirements-dev.txt            Dev/test extras
implementation_plan.md          Roadmap
```

## Run the product

## Installation

Prerequisites: **Python 3.11+** (the repo targets 3.13) and **Node.js 18+**.

```powershell
# Backend
cd app/backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ..\frontend
npm install
```

A repository-level `venv/` may already exist from earlier work; you can reuse it
if it already has the requirements installed.

## Environment configuration

Everything works with sensible defaults, so no environment file is strictly
required for local development.

- **Backend** — copy `app/backend/.env.example` to `.env` (or set the variables in
  your shell / process manager) and uncomment what you need. The names are the
  existing `LVA_*` variables:

  | Variable | Default | Purpose |
  |---|---|---|
  | `LVA_PROJECT_ROOT` | folder above `app/backend` | Repository root |
  | `LVA_MODEL_PATH` | `runs/.../license_plate_detector/weights/best.pt` | YOLO weights |
  | `LVA_BATCH_CSV` | `batch_results/batch_results.csv` | Batch rows |
  | `LVA_DASHBOARD_JSON` | `batch_results/analytics/dashboard_data.json` | Batch aggregates |
  | `LVA_VIDEO_DETECTIONS_CSV` | `video_results/detections.csv` | Video detections |
  | `LVA_VIDEO_SUMMARY_JSON` | `video_results/video_summary.json` | Video summary |
  | `LVA_ANNOTATED_VIDEO` | `video_results/annotated_video.mp4` | Annotated player |
  | `LVA_TRAINING_RESULTS_CSV` | `runs/.../results.csv` | Training metrics |
  | `LVA_FRONTEND_ORIGIN` | localhost:5173 | CORS allow-list (comma-separated) |
  | `LVA_DEBUG_TIMING` | `1` | Per-request timing logs (set `0` in prod) |
  | `LVA_DEBUG_MODE` | `0` | Enables the debug crop endpoint (never in prod) |

- **Frontend** — copy `app/frontend/.env.example` to `.env.local`. Only
  `VITE_API_BASE` is used; it is inlined at **build time**, so changing it requires
  a rebuild. In development it defaults to `http://localhost:8000`; in a production
  build it must be set or every request fails fast with a clear configuration error.

No secrets are required by this project, and the example files contain placeholders
only.

## Running the backend

```powershell
cd app/backend
.\venv\Scripts\activate        # if not already active
uvicorn main:app --reload --port 8000
```

Then check:

- `http://localhost:8000/api/health` — should report `"status": "online"` and, per
  source, `"exists": true` where the CSV/JSON files (and weights/video) are present.
- `http://localhost:8000/api/readiness` — `ready` once models have warmed up.
- `http://localhost:8000/docs` — interactive API documentation.

## Running the frontend

```powershell
cd app/frontend
npm run dev
```

Open `http://localhost:5173`. It talks to the backend at `http://localhost:8000` by
default (override with `VITE_API_BASE`). For a production build:

```powershell
npm run build      # outputs app/frontend/dist
npm run preview     # serve the built bundle locally
```

## Testing & verification

The repository ships unit tests and two end-to-end verification harnesses.

```powershell
# 1. Unit tests (standard library unittest — no extra deps)
venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. Image-detection core: schema, singleton loading, concurrency, memory
venv\Scripts\python.exe verify_suite.py

# 3. Live API: spawns uvicorn and checks health, readiness, detection,
#    multi-plate, two-line plates and event-loop responsiveness under load
$env:LVA_DEBUG_TIMING='0'
venv\Scripts\python.exe verify_live_api.py

# 4. Frontend production build
cd app\frontend
npm run build
```

Developer extras (`requests`, `psutil`) are listed in `requirements-dev.txt`:

```powershell
pip install -r requirements-dev.txt
```

The unit tests cover the detection-geometry layer, the format-repair/validation
logic and the history service. `verify_suite.py` asserts that the image-detection
response keeps exactly its original top-level keys (additive per-plate fields only),
that models load exactly once under a concurrent load race, and reports memory use.
`verify_live_api.py` starts a real server and checks the public behaviour end to end.

> `verify_suite.py` and `verify_live_api.py` load YOLO + EasyOCR, so they need
> `best.pt` on disk and take noticeably longer than the unit tests.

## Model / weights requirements

- **Detector** — Ultralytics YOLO single-class license-plate model.
- **Weights** — `runs/detect/models/license_plate_detector/weights/best.pt`
  (about 5 MB). `*.pt` is gitignored, so you must supply your trained weights at
  that path (or point `LVA_MODEL_PATH` at them) before live detection will work.
- **OCR** — EasyOCR (English), loaded in recogniser-only mode; its model files
  download automatically on first use if they are not already cached.
- **Compute** — CPU-only is supported and is the default; thread pools are capped
  and models are warmed up once at startup.

Copy your trained weights into place:

```powershell
# place best.pt at:
runs\detect\models\license_plate_detector\weights\best.pt
```

## Known limitations

- No authentication — this is a local/demo application. Do not expose the API to
  the public internet without adding access control.
- "OCR success" figures count runs where **some** text was extracted; they are not
  verified character-level accuracy.
- The dashboard's batch and video numbers come from **pre-generated** files; the
  app does not re-run the 658-image batch or the 4K video on load.
- The absolutely-verified training metrics are on the same labeled split used for
  checkpoint selection, so they are slightly optimistic; they are valid for A/B
  comparison, not as a held-out benchmark.
- Large weights/video are gitignored, so a fresh clone needs them supplied
  locally (see above).
- The frontend production bundle is a single large JS chunk (a Vite size warning);
  see `implementation_plan.md` for the planned code-splitting step.
- Only Indian registration formats are validated; other countries are out of scope.

## Future improvements

These are **not yet implemented** — they are the roadmap, tracked in
`implementation_plan.md`:

- Route-level code splitting for the frontend bundle.
- A 404 route and a top-level React error boundary.
- Direct Cloudinary/Azure deployment with persistent storage for weights/results.
- Playwright-based end-to-end UI tests.
- Optional containerisation (Dockerfile) for reproducible deployment.
- An expanded curated sample gallery and richer per-plate detail views.

## Deployment (Render)

Two services from the same repository root:

- **Backend** — build `pip install -r app/backend/requirements.txt`, start
  `cd app/backend && uvicorn main:app --host 0.0.0.0 --port $PORT`. Ship
  `best.pt` with the image (or a persistent disk), or set `LVA_MODEL_PATH`.
- **Frontend** — static site with build `cd app/frontend && npm ci && npm run build`
  and publish directory `app/frontend/dist`. Set `VITE_API_BASE` at **build time**.

| Variable | Required | Purpose |
|---|---|---|
| `LVA_FRONTEND_ORIGIN` | **Yes** | Exact frontend origin(s), comma-separated. Missing/wrong value = browser CORS failures on every call. |
| `VITE_API_BASE` | **Yes** | Full backend origin, inlined at build time. |
| `LVA_MODEL_PATH`, `LVA_PROJECT_ROOT` | No | Override weights / repo layout on the image. |
| `LVA_DEBUG_TIMING`, `LVA_DEBUG_MODE` | No | Set both to `0` in production. |

`GET /api/health` (unchanged schema) reports liveness; `GET /api/readiness`
reports `ready` / `warming_up` / `error` for health checks and the UI warm-up notice.

## License / privacy

This is a local demo: no authentication. Do not expose the API to the public
internet without access control. Training images and the 4K video are not intended
for a public git clone.
