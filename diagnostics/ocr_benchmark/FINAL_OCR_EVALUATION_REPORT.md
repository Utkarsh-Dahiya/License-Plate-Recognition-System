# Final OCR Evaluation — Dataset Readiness Report

**Status: BENCHMARK BLOCKED — insufficient verified ground truth.**

This is a *readiness* report, not a comparative-accuracy report. The comparative
OCR benchmark was **intentionally not run** because the dataset does not yet
meet the minimum evidence threshold. No accuracy numbers below are invented;
they are reported as *unavailable*.

Generated from a live re-scan of the actual files on disk (not summary
counters). Method: `python diagnostics/ocr_benchmark/recalc_counts.py`.

---

## 1. Why the benchmark is blocked

The comparative benchmark requires **>= 50 valid, manually verified, readable
ground-truth plates**. Current count: **2**.

An entry is only counted when the actual record confirms all of: non-empty
ground-truth text, human verification, readability, and passing validation.
OCR predictions, OCR "hints", and model-generated text are **never** counted as
ground truth. Pending and unreadable entries do **not** count.

```
=== READY FOR >=50 VERIFIED BENCHMARK: NO (have 2) ===
```

Running EasyOCR-vs-alternatives on 2 samples would produce noise, not signal,
and would risk selecting a model on anecdote. That is explicitly disallowed by
the task. The blocker is therefore **human transcription work**, not a code
problem.

---

## 2. Dataset counts (recomputed from records)

| Metric | Value |
|--------|-------|
| Total candidate entries | **62** |
| Verified readable ground truth | **2** |
| Pending transcription | **60** |
| Marked unreadable | **0** |
| Missing crops | **0** |
| Duplicate / invalid sample IDs | **0** |

The two verified entries are exactly:

| ID | Ground truth | Verified |
|----|--------------|----------|
| `V0034` | `9M17408FV1232` | true |
| `V0333` | `EHK1642` | true |

All 62 crop paths resolve to real files; no expected crop is missing.

---

## 3. Comparative OCR metrics

| Metric | Status |
|--------|--------|
| Exact-match accuracy | Not computed — n < 50 |
| Character-level accuracy | Not computed — n < 50 |
| OCR failure rate | Not computed — n < 50 |
| Average / P50 / P95 latency | Not computed — blocked |
| Single-line vs two-line performance | Not computed — blocked |
| Performance by resolution | Not computed — blocked |
| Model init time / memory | Not computed — blocked |

No model is recommended or selected, because no controlled measurement exists.
`EasyOCR` is currently the production baseline and remains so until evidence
justifies a change.

---

## 4. Known failure case (GJ08DJ3136)

Reported example: actual `GJ08DJ3136`, incorrect OCR `76J080J3136`.

**This case could not be tested.** A repository-wide scan found the string only
in prose — the narrative of `BENCHMARK_REPORT.md` and a note in `results.json`.
There is **no** matching image crop and **no** verified ground-truth label for
it in `ground_truth_readable.json` or any data file. It is therefore an
**unverified historical example**, not an evaluable sample, and no success is
claimed for it.

---

## 5. What to do to unblock (exact steps)

1. Open `diagnostics/ocr_benchmark/transcribe.html` in a browser (double-click
   the file, or VS Code → *Open with Live Server*).
2. Work through the **60 Pending** entries. For each crop you can read:
   type the **exact** text → **Save** (`Ctrl+S`) → **Verify** (`V`). For plates
   you truly cannot read, press **Unreadable** (`U`). Use **Review** (`R`) for
   anything uncertain.
3. Press **Export answers JSON** and save the download.
4. Back up and apply (from the repo root, PowerShell):

   ```powershell
   $ans = "diagnostics\ocr_benchmark\review_sheets\transcript_answers.json"
   Copy-Item $ans "$ans.bak" -Force
   Copy-Item "$env:USERPROFILE\Downloads\transcript_answers.json" $ans -Force
   python diagnostics\ocr_benchmark\apply_transcript.py
   python diagnostics\ocr_benchmark\recalc_counts.py
   ```

5. Repeat until **verified readable GT >= 50**. You currently need **48 more**
   verified readable plates.

Once the gate is met, re-run the controlled benchmark (EasyOCR baseline vs
justified alternatives) on identical crops and ground truth, and replace this
readiness report with real metrics in `final_ocr_results.json` /
`final_ocr_per_sample.json`.

---

## 6. Production protection

This task did not modify `detection_service.py`, the YOLO weights
(`best.pt`), the YOLO confidence threshold (`_DEFAULT_CONF = 0.20`), production
OCR preprocessing/ranking/normalization/fallback, the API, the frontend, or the
Render deployment config. All work stays under `diagnostics/ocr_benchmark/`.
