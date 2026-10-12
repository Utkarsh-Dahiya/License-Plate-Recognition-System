@echo off
REM ============================================================================
REM  Plate Transcription Reviewer launcher
REM  Opens the offline browser-based ground-truth transcription tool.
REM  Scope: diagnostics only. This does NOT touch production code, the YOLO
REM  model, the confidence threshold, the OCR pipeline, the API, or the frontend.
REM ============================================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   License Plate Transcription Reviewer
echo ============================================================
echo.
echo  Opening transcribe.html in your default browser...
echo.
echo  QUICK STEPS:
echo    1. Read the crop, then type the EXACT plate text in the box.
echo    2. Press Save (Ctrl+S) to keep it, then Verify (V) when certain.
echo       (U) = mark unreadable   (R) = flag for another review.
echo    3. Click "Export answers JSON" and save the download.
echo    4. Apply + recount from PowerShell (repo root):
echo.
echo         $ans = "diagnostics\ocr_benchmark\review_sheets\transcript_answers.json"
echo         Copy-Item $ans "$ans.bak" -Force
echo         Copy-Item "$env:USERPROFILE\Downloads\transcript_answers.json" $ans -Force
echo         python diagnostics\ocr_benchmark\apply_transcript.py
echo         python diagnostics\ocr_benchmark\recalc_counts.py
echo.
echo  The OCR "hint" shown on screen is a PREDICTION, not ground truth.
echo  Only what you type and Verify counts. Nothing is auto-filled.
echo ============================================================
echo.

start "" "%~dp0transcribe.html"
endlocal
