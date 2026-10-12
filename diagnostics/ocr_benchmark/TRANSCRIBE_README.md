# Plate Transcription Reviewer — how to use

A small, offline, browser-based tool for **manually reading the diagnostic
plate crops** and entering **ground-truth text** for the OCR benchmark.
It is a ground-truth collection tool: it never treats OCR output as truth and
never auto-fills your answers.

> Scope: this lives entirely under `diagnostics/ocr_benchmark/` and does **not**
> touch production code, the YOLO model, the confidence threshold, the OCR
> pipeline, the API, or the frontend.

## Files

| File | Role |
|------|------|
| `transcribe.html` | The UI. Open this in a browser. |
| `transcribe.js` | UI logic (vanilla JS, no framework). |
| `transcribe_data.js` | Generated dataset (entries, crop paths, OCR hints). |
| `gen_data_js.py` | Regenerates `transcribe_data.js` from `ground_truth_readable.json`. |
| `apply_transcript.py` | Merges your exported answers into `ground_truth_readable.json`. |

## Launch (Windows, from VS Code)

No server needed. Either:

1. In VS Code Explorer, right-click `diagnostics/ocr_benchmark/transcribe.html`
   → **Reveal in File Explorer**, then double-click it (opens in your default
   browser via `file://`), **or**
2. Right-click the file → **Open with Live Server** (VS Code Live Server
   extension), if you have it.

Both work. `file://` is simplest.

## Workflow

1. The list starts on **Pending** entries (60 of them). Two entries,
   `V0034` and `V0333`, are already verified and shown **locked (read-only)**.
2. Read the crop on the left. Use the image tools to inspect it:
   **Zoom +/−, Fit, Reset, Brightness, Contrast, Grayscale**. These only change
   the on-screen image — the crop file on disk is never modified.
3. Type the **exact** plate text into the box and press **Save** (`Ctrl+S`, or
   `Enter` to save and advance). Your text is stored **exactly** as typed.
4. Mark **Unreadable** (`U`) for plates you truly cannot read, or **Review**
   (`R`) to flag one for a second pass.
5. Press **Verify** (`V`) only when you are certain — this is the explicit human
   action that makes it count as verified ground truth.
6. When done (or periodically), press **Export answers JSON**. Save the file as
   `diagnostics/ocr_benchmark/review_sheets/transcript_answers.json`
   (overwriting the blank template). Use **Import JSON** to reload it later or
   on another machine.
7. **Back up the current answers file before replacing it**, then move your
   export into place and apply (run from the repo root):

   ```powershell
   $ans = "diagnostics\ocr_benchmark\review_sheets\transcript_answers.json"
   Copy-Item $ans "$ans.bak" -Force
   Copy-Item "$env:USERPROFILE\Downloads\transcript_answers.json" $ans -Force
   python diagnostics\ocr_benchmark\apply_transcript.py
   ```

   `apply_transcript.py` writes verified, readable entries back into
   `ground_truth_readable.json` and prints the total verified-readable count.
8. Re-check the authoritative count (computed from individual records, not
   summary counters):

   ```powershell
   python diagnostics\ocr_benchmark\recalc_counts.py
   ```

   Confirm the **verified readable GT** number. The comparative OCR benchmark
   unlocks only at **>= 50** verified readable plates.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next entry |
| `S` or `Ctrl+S` | Save |
| `Enter` | Save and advance |
| `U` | Mark unreadable |
| `R` | Mark review-needed |
| `V` | Mark verified |
| `+` / `-` | Zoom in / out |
| `0` | Fit to screen |

Shortcuts are **disabled while typing** in the text box (except `Ctrl+S` /
`Ctrl+Enter`, which still save). Progress auto-saves to the browser's local
storage, so reopening the page restores where you left off.

## Verified-entry protection

`V0034` (`9M17408FV1232`) and `V0333` (`EHK1642`) are locked by default. To
change one, click **Modify verified** on that entry, confirm the overwrite in
the dialog, edit, and Save. The overwrite also requires a second confirmation.
Until you do that, their verified values are preserved exactly and are exported
unchanged.

## Missing crops

If a crop file is missing, the viewer shows a clear "Could not load crop"
message with the expected path instead of failing silently. The dataset lists
which entries are affected; use the filters to skip them.

## Regenerating after `ground_truth_readable.json` changes

```bat
python diagnostics\ocr_benchmark\gen_data_js.py
```

Then reload `transcribe.html`. This never modifies your saved answers.
