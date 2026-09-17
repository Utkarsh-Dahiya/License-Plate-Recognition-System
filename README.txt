# License Vision AI — Image + Video Testing (offline scripts)

These notes describe the **root Python scripts**, not the FastAPI/React app in `app/`.
The dashboard does **not** re-run these scripts on load.

## Batch image testing

From this project folder:

```powershell
python batch_test.py
```

There is **no** `--input` / `--output` CLI. Paths are hardcoded in `batch_test.py`:

- Model: `runs/detect/models/license_plate_detector/weights/best.pt`
- Images: `YOLO_dataset/images` (note the folder name in the script)
- Output: `batch_results/` including `batch_results.csv`

Then `batch_results/analytics.py` can rebuild `batch_results/analytics/dashboard_data.json` from that CSV.

The batch script reports detection rate and OCR **confidence / whether text was extracted**. It does **not** claim OCR accuracy unless ground-truth plate labels are supplied.

## Video ALPR

`video_pipeline.py` has **no** `--video` CLI. Paths are hardcoded:

- Input: `test_video.mp4` (project root)
- Model: `runs/detect/models/license_plate_detector/weights/best.pt`
- Output directory: `video_results/`

The **current script** writes consensus filenames:

- `video_results/annotated_video_consensus.mp4`
- `video_results/detections_consensus.csv`
- `video_results/video_summary_consensus.json`

The **dashboard and API** read the existing files already on disk (do not regenerate the 4K run):

- `video_results/annotated_video.mp4`
- `video_results/detections.csv`
- `video_results/video_summary.json`

Do not re-run the 4K source to “refresh” the product. The live app’s “Process a new video” path is a separate demo job and does not replace those files.

## App (product)

See `README.md` for FastAPI + React setup.
