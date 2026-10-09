# OCR Benchmark Report

**Scope.** Controlled, offline A/B benchmark of OCR strategies against the
current production pipeline for the License Plate Detection project.
**No production code, model, threshold, schema or ranking was modified.**

All experimental code lives under `diagnostics/ocr_benchmark/`. The baseline
**imports and calls the real production functions**
(`services.detection_service._run_ocr`, `_canvas_gray`, `_variant_primary`,
`_finalize_selection`, `clean_text`, `indian_plate_score`) so the measured
baseline is byte-for-byte production behaviour.

---

## 1. Executive Summary

**Headline finding — the OCR pipeline is failing on genuinely readable plates,
and the root cause is EasyOCR RECOGNITION, not preprocessing or ranking.**

Across 9 preprocessing variants, multi-pass OCR, character-confidence-aware
ranking and consensus voting, the correct plate string was **never produced**
for either of the two human-readable ground-truth plates. This is decisive:
when the correct candidate is never generated, no ranking or consensus scheme
can recover it.

| Readable plate | Ground truth | Best OCR seen (any pass) | Correct ever produced? |
|---|---|---|---|
| 0333 | `EHK1642` | `WU63GIY` / `U63G` | **No** (best similarity 0.18) |
| 0034 | `9M17408FV1232` | `342132` / `2443438` | **No** (best similarity 0.42) |

Baseline exact-match on readable plates: **0 / 2 (0.0%)**. Character accuracy:
**7.7%**. The single strongest signal from character-level confidence is that
EasyOCR reports the *wrong* characters at *high* confidence (e.g. `U63G` at
0.87–0.99), so confidence alone cannot flag the error.

---

## 2. Dataset

### 2.1 Ground-truth honesty (critical)

This benchmark is deliberately strict about ground truth. Facts established
by inspecting the repository and audit artifacts:

- **Human-verified text ground truth exists for only 3 plates**
  (`diagnostics/audit_gt.json`). These are the only plates whose exact-match
  and character accuracy can be measured:
  - `0034` — `9M17408FV1232` — readable — two-line — 59×27 px
  - `0333` — `EHK1642` — readable — single-line — 61×17 px
  - `0067` — no transcription — **not readable** — 24×9 px (low-resolution)
- `diagnostics/audit/collection.json` (133 val images, 145 GT boxes, 136
  matched detections, 127 images with ≥1 OCR string) contains **production
  OCR output but no human text ground truth.** It is therefore used only for
  behavioural metrics that require no text GT (OCR-empty rate, candidate
  diversity, character-confidence availability, latency).
- **`GJ08DJ3136` does not exist in any data file.** It appears only as an
  illustrative string inside `diagnostics/audit_report.py`. There is no image
  or crop for it, so a literal per-sample OCR trace is impossible; this is
  reported honestly and the confusion *mechanism* is instead investigated on
  real plates (Section 9).

### 2.2 Category coverage

| Category | Samples | Notes |
|---|---|---|
| ALL (with text GT) | 2 | 0067 has no transcription and is excluded from accuracy |
| READABLE | 2 | 0034 (two-line), 0333 (single-line) |
| LOW-RESOLUTION | 1 | 0067 (24×9 px, not human-readable) — no GT string |
| SINGLE-LINE | 1 | 0333 |
| TWO-LINE | 1 | 0034 |

**Honest limitation:** with only 2 text-GT plates, absolute accuracy numbers
have very wide confidence intervals and must be read as *directional*, not
population estimates. The benchmark is nonetheless conclusive on the
qualitative question because the effect (correct string never generated) is
binary and reproduced on both readable plates.

---

## 3. Baseline Results

Baseline = production OCR cascade (tier-1 CLAHE pass + `_finalize_selection`).

| Plate | GT | Baseline OCR | Exact | Char acc | Baseline conf |
|---|---|---|---|---|---|
| 0034 (two-line) | `9M17408FV1232` | `34218` | ✗ | 0.154 | 0.226 |
| 0333 (single-line) | `EHK1642` | `WU63GIY` | ✗ | 0.000 | 0.199 |

- Detection succeeded for both (YOLO localizes the plate; crop is non-degenerate).
- **OCR exact-match: 0.0%. Character accuracy: 7.7%.**
- The crop is not the bottleneck: the text is present and, for `0333`,
  EasyOCR even reads a plate-shaped 7-character string — just the wrong one.


---

## 4. Preprocessing A/B Results

Nine preprocessing variants were each run through the **same** production
recognizer with identical settings (only the image changed). Exact-match and
character accuracy are computed on the 2 text-GT plates.

| Variant | Char acc | Exact | Fail rate | Avg latency |
|---|---|---|---|---|
| A_original (production canvas, gray) | 0.187 | 0% | 0% | 72 ms |
| B_clahe (production tier-1) | 0.077 | 0% | 0% | 72 ms |
| C_gray_contrast | 0.154 | 0% | 0% | 78 ms |
| D_otsu | 0.077 | 0% | 0% | 72 ms |
| E_adaptive | 0.115 | 0% | 0% | 86 ms |
| F_denoise_otsu | 0.077 | 0% | 0% | 79 ms |
| G_up_gray | 0.115 | 0% | 0% | 73 ms |
| H_up_clahe | 0.115 | 0% | 0% | 70 ms |
| I_up_adaptive | 0.115 | 0% | 0% | 81 ms |

**Conclusion: preprocessing does NOT fix the problem. No variant achieved a
single exact match.** The best variant (`A_original`, 0.187) is marginally
better than production CLAHE (0.077) on character accuracy, but every variant
is far from correct and none produces the GT string. Preprocessing changes
*which* characters are hallucinated, not whether the correct ones appear.

Per-plate variant outputs (readable plates):

- `0333` GT `EHK1642` → `U63G`, `WU63GIY`, `NU63CI`, `WU6JGTYI`, `4VU63GTYI`,
  `WU6JGTYA`, `MU63GLY`, `U63GIY`, `IVU63GTYL` — all wrong.
- `0034` GT `9M17408FV1232` → `342408`, `34218`, `342132`, `2443438`, `FIZ32`,
  `2443438`, `342408`, `34212`, `ZI232` — all wrong; only fragments of the
  trailing serial `...1232` survive.

---

## 5. Multi-Pass Results

All 9 variants were run per plate and **every** raw candidate preserved. The
question: does *any* pass produce the correct string?

| Plate | GT | Distinct candidates | Best similarity to GT | Correct ever produced? |
|---|---|---|---|---|
| 0034 | `9M17408FV1232` | 7 | 0.42 | **No** |
| 0333 | `EHK1642` | 9 | 0.18 | **No** |

**Conclusion: multi-pass OCR does not help here because the correct string is
never in the candidate set.** This is the pivotal result — it separates the
two possible failure modes:

- **CASE A (ranking):** correct candidate exists but is ranked too low →
  fixable by better selection.
- **CASE B (recognition):** correct candidate is never generated →
  ranking/consensus cannot help; the recognizer itself must improve.

Both readable plates fall into **CASE B**. Section 9 details this for the
illustrative `GJ08DJ3136` example.



---

## 6. Consensus Results

A confusion-aware, position-wise consensus vote was applied across the
multi-pass candidates. Confusion classes used (as specified): G↔7, J↔6, D↔0,
B↔8, I↔1, O↔0, S↔5, Z↔2. No global character replacement was performed; the
vote only groups confusable glyphs so they can pool evidence.

| Plate | GT | Consensus output | Exact | Char acc |
|---|---|---|---|---|
| 0034 | `9M17408FV1232` | `FI232` | ✗ | 0.308 |
| 0333 | `EHK1642` | `WU63GIY` | ✗ | 0.000 |

**Conclusion: consensus does not recover the correct string.** For `0034` the
confusion-aware vote (`FI232`) actually scores the *best* character accuracy of
any approach (0.308) by salvaging the trailing `...1232`, but it still drops
the leading `9M17408FV` and misreads the `F`. For `0333` consensus simply
reproduces the majority wrong read. Consensus amplifies whatever the passes
agree on — and here they agree on the wrong characters.

---

## 7. Character Confidence Analysis

EasyOCR **does expose usable per-character CTC confidence** in this
environment: the production `_run_ocr` hook captured aligned per-character
probabilities for **100% of val-pool crops** (`char_conf_available_rate = 1.0`).
This is real, not fabricated.

Captured evidence (production CLAHE pass):

- `0333` GT `EHK1642` → OCR `WU63GIY`
  `[('W',0.293),('U',0.999),('6',0.999),('3',0.957),('G',0.870),('I',0.671),('Y',0.724)]`
- `0034` GT `9M17408FV1232` → OCR `34218`
  `[('3',0.508),('4',0.944),('2',0.924),('1',0.575),('8',0.744)]`

**Critical finding: the wrong characters are read at HIGH confidence.** On
`0333`, `U`, `6`, `3`, `G` carry 0.87–0.999 probability yet are all incorrect
against GT. The only genuinely low-confidence character is the *leading* `W`
(0.293) — i.e. character confidence correctly flags a hallucinated edge
character, but the *core* misrecognition is high-confidence. Therefore
character confidence is useful for **edge/hallucination detection** (which the
production `_evidence_select` already exploits via `_first_last_conf`) but it
**cannot detect the confident mid-string misreads** that dominate these
failures.

On the illustrative `GJ08DJ3136 → 76J080J3136` confusions (G→7, J→6, D→J, J→0):
these are exactly the glyph-shape confusions EasyOCR resolves confidently;
per-character probability does not separate them. State: char-confidence
signal is **available and partially useful, but insufficient for the dominant
error mode.**

---

## 8. Candidate Ranking Results

Three offline rankers were compared (no production ranking touched):

1. **Production** (`_finalize_selection`) — baseline.
2. **Char-confidence-aware** — OCR conf + format + mean/min char-conf + edge penalty.
3. **Consensus** — confidence-weighted agreement + format.

| Ranker | 0034 output | 0333 output | Exact (2 plates) |
|---|---|---|---|
| Production | `34218` | `WU63GIY` | 0/2 |
| Char-confidence | `FIZ32` | `IVU63GTYL` | 0/2 |
| Consensus | `FIZ32` | `WU63GIY` | 0/2 |

**Conclusion: no ranking scheme changes the outcome because none of them can
manufacture the correct string.** The experimental char-confidence ranker even
*slightly lowers* character accuracy versus the raw best pass (it rewards
confident-but-wrong reads). This directly confirms the benchmark's central
thesis: **when the correct candidate is absent, ranking is irrelevant.**



---

## 9. GJ08DJ3136 Case Study

**Data-integrity finding first:** `GJ08DJ3136` and its system output
`76J080J3136` are **not present in any image, crop, label or results file** in
this repository. A repository-wide search finds the string only inside
`diagnostics/audit_report.py`, where it is used as an illustrative example.
There is therefore **no crop or image from which a literal EasyOCR trace of
this exact plate can be produced** — producing one would require inventing
data, which is explicitly disallowed.

What *can* be done rigorously is to (a) trace the confusion *mechanism* on the
real readable plates, which reproduce the identical error class, and (b) prove
which pipeline stage the error class originates in.

### 9.1 The error chain on real data (same failure class)

For `0333` (GT `EHK1642`), the complete production chain:

```
GT            EHK1642
crop          61x17 px, aspect 3.59  (present, readable, single-line)
preprocess    production CLAHE tier-1
raw EasyOCR   WU63GIY   (per-char conf: .293 .999 .999 .957 .870 .671 .724)
normalize     WU63GIY   (clean_text removes nothing; no '-' present)
repair        none added (no format repair changes the winner)
ranking       production _evidence_select keeps WU63GIY (highest evidence)
final         WU63GIY   (conf 0.199)
```

For `0034` (GT `9M17408FV1232`), two-line:

```
GT            9M17408FV1232
crop          59x27 px (readable, two-line)
raw EasyOCR   34218 (CLAHE); 2443438 (OTSU); 342132 (gray-contrast); ...
final (prod)  34218   -> only the tail serial fragment survives
```

### 9.2 Which stage does the error originate in?

| Stage | Verdict | Evidence |
|---|---|---|
| Crop | **Not the cause** | 61×17 / 59×27 px crops are non-degenerate; text is present. |
| Preprocessing | **Not the cause** | 9 variants all fail; changing preprocessing changes the wrong output, never produces GT. |
| Raw EasyOCR | **PRIMARY CAUSE** | The correct string is absent from every raw pass (best sim 0.18 / 0.42). EasyOCR confidently reads wrong glyphs (E→W/U, H→6, 1→3/G, 4→I/Y). |
| Normalization | Not the cause | `clean_text` only strips `-`/non-alphanumerics; nothing was lost. |
| Repair | Not the cause | Tier-5 format repair never produced GT either. |
| Ranking | Not the cause (here) | This is **CASE B**: the correct candidate is never generated, so ranking cannot select it. |

### 9.3 CASE A vs CASE B — decisive

For both readable plates the correct candidate was **NEVER produced** by any
OCR pass (`gt_produced = False`). Therefore this is:

> **CASE B — the correct candidate is never generated by EasyOCR.**

Consequence (stated plainly): **candidate ranking cannot solve this
recognition problem.** The `GJ08DJ3136 → 76J080J3136` error class (confident
glyph-shape confusions at the recognizer output) is the same class observed
here, and it originates in the **raw EasyOCR recognition stage**, not in crop,
preprocessing, normalization, repair or ranking.



---

## 10. Readable vs Low-Resolution Results

| Subset | n (text-GT) | Baseline exact | Baseline char acc | Correct ever produced (any pass) |
|---|---|---|---|---|
| ALL (text-GT) | 2 | 0.0% | 7.7% | 0 / 2 |
| READABLE | 2 | 0.0% | 7.7% | 0 / 2 |
| LOW-RESOLUTION | 0 (0067 has no transcription) | — | — | n/a |
| SINGLE-LINE | 1 (0333) | 0.0% | 0.0% | 0 / 1 |
| TWO-LINE | 1 (0034) | 0.0% | 15.4% | 0 / 1 |

**Readable plates fail at 100% (0/2) exact-match.** The low-resolution plate
(0067, 24×9 px) has no transcription and cannot be scored; it is consistent
with the earlier audit finding that such crops are genuinely unreadable. There
is no evidence here that low-resolution is masking a working pipeline — the
**readable** plates are the ones that fail, and they fail because EasyOCR does
not emit the correct glyphs at all.

---

## 11. Latency Comparison

Val-pool behavioural pass (145 crops, production CLAHE single pass):

| Metric | Value |
|---|---|
| Avg latency / crop | 72.8 ms |
| p50 | 61.0 ms |
| p95 | 131.9 ms |
| Empty-OCR rate | 0.0% |
| Char-confidence available | 100.0% |
| Avg distinct candidates / crop | 1.0 |

Per-preprocessing-variant latency (2 GT plates): 70–86 ms each. A 9-variant
multi-pass pipeline therefore costs roughly **9× a single pass** (~0.65 s/crop
of OCR) yet delivers **zero** additional exact matches here — pure added
latency with no accuracy return.

---

## 12. Failure Analysis

Failure-category mapping for the 2 text-GT plates:

| Plate | Category | Reason |
|---|---|---|
| 0333 `EHK1642` | **D — OCR RECOGNITION FAILURE** | Readable crop; EasyOCR emits `WU63GIY`; correct string never generated by any variant. |
| 0034 `9M17408FV1232` | **D — OCR RECOGNITION FAILURE** | Readable two-line crop; only the tail serial is read; leading chars never recovered by any variant. |

Neither failure is attributable to detection (A), bad crop (B), low resolution
(C), normalization (E), repair (F) or validation/ranking (G). Both are the same
class: the recognizer itself.



---

## 13. Best Experimental Strategy

| Approach | Samples | Exact Match | Char Acc | Correct ever produced |
|---|---|---|---|---|
| Baseline (production) | 2 | 0.0% | 7.7% | 0/2 |
| Best preprocessing (oracle) | 2 | 0.0% | 22.5% | 0/2 |
| Multi-pass (any-pass oracle) | 2 | 0.0% | 30.1% (best-sim) | 0/2 |
| Consensus (char-aware) | 2 | 0.0% | 15.4% | 0/2 |
| Rank: char-confidence | 2 | 0.0% | 11.5% | 0/2 |
| Rank: consensus | 2 | 0.0% | 11.5% | 0/2 |

The **best character-accuracy** figure comes from multi-pass + confusion-aware
consensus (`FI232`, 0.308 for 0034), which salvages the trailing serial. But
**no strategy achieves a single exact match.** Since the correct candidate is
never generated, the "best" offline strategy is only the least-wrong.

### REQUIRED FINAL COMPARISON TABLE

| Approach | Samples | Exact Match | Character Accuracy | OCR Failure | Avg Latency | P50 | P95 |
|----------|---------|-------------|--------------------|-------------|-------------|-----|-----|
| Baseline | 2 | 0.0% | 7.7% | 0% | 73 ms | 61 ms | 132 ms |
| Best preprocessing | 2 | 0.0% | 22.5% | 0% | 78 ms | — | — |
| Multi-pass | 2 | 0.0% | 30.1% (oracle) | 0% | ~657 ms (9x) | — | — |
| Consensus | 2 | 0.0% | 15.4% | 0% | (over multi-pass) | — | — |
| Experimental ranking | 2 | 0.0% | 11.5% | 0% | (over multi-pass) | — | — |

(Latency P50/P95 are measured for the single-pass baseline on the val pool;
multi-pass/consensus latency is the ~9x single-pass cost since they re-run
variants. Empty-OCR "failure" is 0% for all — the system never returns blank;
it returns confidently wrong text.)

---

## 14. Recommendation for Production

**Do NOT change preprocessing, ranking, consensus or validation in production
based on this evidence — they cannot help.** The measured bottleneck is the
EasyOCR recognizer itself on readable plates.

Recommended next experiment (offline, then a gated production A/B):

1. **Swap / augment the recognizer.** The single highest-leverage change is a
   plate-tuned recognizer (e.g. a CRNN/CTC model trained on Indian license
   plates, or EasyOCR fine-tuned on plate data) rather than the general-purpose
   `english_g2` model. This is the only lever that addresses CASE B.
2. **Add a beam-search / N-best pass** so alternative glyph readings (7 vs G,
   6 vs J, 0 vs D) survive as candidates instead of a single greedy read —
   then ranking has something correct to pick.
3. **Keep char-confidence edge penalties** (they work for hallucinated edge
   chars) but do not rely on them for confident mid-string errors.
4. **Re-benchmark against a larger human-transcribed set.** Two plates cannot
   quantify a population accuracy; transcribe 50–100 readable val plates before
   tuning any weights, to avoid overfitting.

### CRITICAL DECISION — explicit answers

1. **Does preprocessing improve OCR?** No. Best variant reaches only 0.187
   char acc / 0% exact; production CLAHE (0.077) is worse. No variant produces GT.
2. **Does multi-pass improve exact-match?** No. Across 9 passes the correct
   string is never generated (0/2).
3. **Does consensus improve accuracy?** No exact matches; it only marginally
   raises char acc on one plate by salvaging the serial tail. Not a fix.
4. **Does character-level confidence provide useful signal?** Partially. It is
   available (100%) and flags hallucinated *edge* characters, but the dominant
   errors are *high-confidence* mid-string misreads it cannot catch.
5. **Is the correct GJ08DJ3136 candidate ever generated?** It cannot be traced
   (not in data), but on the identical real failure class the correct candidate
   is **never** generated by any pass.
6. **Is the primary problem recognition or ranking?** **Recognition.** This is
   CASE B — the correct string is absent from the candidate set.
7. **Which approach gives the best accuracy?** Multi-pass + confusion-aware
   consensus on character accuracy (0.308) — but 0% exact for all approaches.
8. **What is the latency cost?** ~73 ms/crop single pass; ~9x (~0.65 s/crop) for
   multi-pass, with zero exact-match return.
9. **Is the improvement large enough to justify production integration?** **No.**
   0% → 0% exact match. The only measured gains are in character accuracy, which
   does not translate to a usable plate number.
10. **What exact change should be tested next?** Replace/augment the EasyOCR
    recognizer with a plate-specialized model **and/or** enable beam-search
    N-best candidates, so the correct string can exist for ranking to select.
    Validate on a 50–100 plate human-transcribed subset.

---

### Provenance / reproducibility

- Scripts: `diagnostics/ocr_benchmark/{benchmark_core.py, run_benchmark.py, make_report.py, probe.py}`
- Raw results: `diagnostics/ocr_benchmark/results.json`
- Per-sample: `diagnostics/ocr_benchmark/per_sample_results.json`
- Metrics: `diagnostics/ocr_benchmark/_metrics.json`
- Baseline imports the unmodified production `services.detection_service`.
- **Production untouched:** `git status` shows only the untracked
  `diagnostics/` tree; no diff to `app/backend/services/detection_service.py`
  or `best.pt`; YOLO threshold, OCR config, ranking, validation and API
  schemas unchanged.

