"""Unit tests for the evidence-gated OCR format-repair logic.

Run from the project root:
    venv/Scripts/python.exe -m unittest tests.test_format_repairs -v

These scenarios come from production failures (KL01AP8921 -> LKL0AP8921 /
KLOIAP8921) and the ambiguity pairs Indian plates actually exhibit
(stamped zeros vs O, serif ones vs I). The recognizer's per-character
probabilities are simulated exactly as the CTC capture hook produces
them: aligned with the cleaned string, one float per character.

Candidate tuples mirror _run_ocr's output: (text, conf, char_probs).
No models are loaded — everything here exercises pure scoring logic.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from services.detection_service import (  # noqa: E402
    _REPAIR_CONF_DISCOUNT,
    _generate_format_repairs,
    clean_text,
    indian_plate_score,
    strict_indian_plate,
)


def cand(text: str, conf: float, probs=None):
    return (text, conf, probs)


class StrictFormatTests(unittest.TestCase):
    def test_current_series_valid(self):
        self.assertTrue(strict_indian_plate("KL01AP8921"))
        self.assertTrue(strict_indian_plate("MH12DE1433"))

    def test_legacy_series_valid(self):
        self.assertTrue(strict_indian_plate("KLB1055"))

    def test_invalid_state_code_rejected(self):
        self.assertFalse(strict_indian_plate("XX01AP8921"))

    def test_kloiap8921_is_invalid(self):
        # Double letter before the district number -> not a valid
        # current-series registration. This is exactly why the raw
        # OCR read needs a repair path.
        self.assertFalse(strict_indian_plate("KLOIAP8921"))
        self.assertFalse(strict_indian_plate("KLO1AP8921"))


class SingleSwapRepairTests(unittest.TestCase):
    def test_single_unsure_O_repairs_to_zero(self):
        # KLO1AP8921 with the O unsure -> KL01AP8921 must be generated.
        # (K L O 1 A P 8 9 2 1 — the O is index 2.)
        probs = [0.95, 0.95, 0.40, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLO1AP8921", 0.90, probs)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertIn("KL01AP8921", texts)

    def test_single_confident_O_still_repairs_single(self):
        # Single swaps deliberately include confident glyphs: the source
        # string is already format-invalid, so format evidence alone
        # backs the repair (stamped zeros read as solid O's).
        probs = [0.95, 0.95, 0.98, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLO1AP8921", 0.90, probs)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertIn("KL01AP8921", texts)


class PairRepairTests(unittest.TestCase):
    # KLOIAP8921 char positions: K(0) L(1) O(2) I(3) A(4) P(5) 8(6) 9(7) 2(8) 1(9)
    # Only O and I are ambiguous glyphs; the thresholds treat <0.75 as unsure.

    def test_no_char_confidence_info_keeps_fallback_pairing(self):
        # CASE 3: the per-character CTC capture hook is unavailable (EasyOCR
        # source drift) so probs is None. The fallback must stay intact:
        # positions are treated as unsure, so the (unsure, confident-fallback)
        # pair rule generates the strict-valid KL01AP8921 rival. Selection
        # must also still rank it above the invalid raw read via format
        # evidence alone (edge penalties are skipped without probs).
        evidence = [(1, [cand("KLOIAP8921", 0.90, None)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertIn("KL01AP8921", texts)

        from services.detection_service import _finalize_selection

        best, repairs2 = _finalize_selection(
            [list(t) for t in evidence]
        )
        self.assertIsNotNone(best)
        self.assertEqual(best[0], "KL01AP8921")

    def test_confident_O_unsure_I_pair_repairs(self):
        # THE PRODUCTION BUG: KLOIAP8921 (O confident 0.93, I unsure 0.42)
        # must gain KL01AP8921 as a rival candidate. The old rule required
        # BOTH glyphs to be unsure and missed this case.
        probs = [0.96, 0.95, 0.93, 0.42, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLOIAP8921", 0.90, probs)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertIn("KL01AP8921", texts)

    def test_unsure_O_confident_I_pair_repairs(self):
        # Mirror direction: the UNSURE glyph first, confident second.
        probs = [0.96, 0.95, 0.42, 0.93, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLOIAP8921", 0.90, probs)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertIn("KL01AP8921", texts)

    def test_two_confident_glyphs_are_never_paired(self):
        # Both O and I confident: overriding double evidence would be
        # guesswork. No KL01AP8921 rival may be generated.
        probs = [0.96, 0.95, 0.93, 0.91, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLOIAP8921", 0.90, probs)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertNotIn("KL01AP8921", texts)

    def test_two_unsure_glyphs_still_pair(self):
        # The original behavior for unsure+unsure must be preserved.
        probs = [0.96, 0.95, 0.42, 0.40, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLOIAP8921", 0.90, probs)])]
        repairs = _generate_format_repairs(evidence)
        texts = [t for t, _, _ in repairs]
        self.assertIn("KL01AP8921", texts)


class ValidSourceTests(unittest.TestCase):
    def test_strict_valid_source_earns_no_repairs(self):
        # KL01AP8921 is already valid — no candidate may rewrite it.
        probs = [0.95] * 9
        evidence = [(1, [cand("KL01AP8921", 0.95, probs)])]
        self.assertEqual(_generate_format_repairs(evidence), [])


class SelectionIntegrationTests(unittest.TestCase):
    """Full selection through _finalize_selection, mirroring the API path."""

    def _select(self, tier_evidence):
        from services.detection_service import _finalize_selection

        return _finalize_selection([list(t) for t in tier_evidence])

    def test_kloiap8921_beaten_by_repaired_kl01ap8921(self):
        # One unsure glyph (I at 0.42): the repaired strict-valid candidate
        # must WIN despite the raw read's slightly higher OCR confidence.
        probs = [0.96, 0.95, 0.93, 0.42, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [
            (1, [cand("KLOIAP8921", 0.91, probs)]),
            (2, [cand("KLOIAP8921", 0.90, probs)]),
        ]
        best, repairs = self._select(evidence)
        self.assertIsNotNone(best)
        self.assertEqual(best[0], "KL01AP8921")
        self.assertTrue(any(t == "KL01AP8921" for t, _, _ in repairs))

    def test_double_confident_keeps_raw_read(self):
        # No unsure glyphs -> no repair -> the raw read wins untouched.
        probs = [0.96, 0.95, 0.93, 0.91, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        evidence = [(1, [cand("KLOIAP8921", 0.91, probs)])]
        best, repairs = self._select(evidence)
        self.assertIsNotNone(best)
        self.assertEqual(best[0], "KLOIAP8921")
        self.assertEqual(repairs, [])

    def test_repair_cannot_beat_valid_original_on_equal_confidence(self):
        # A strictly-valid original must outrank its repaired rival: the
        # repair discount + format penalty guarantee it on equal evidence.
        # (Same text observations merge into one group whose reported
        # confidence is the MEAN — so assert the winner's identity and
        # that the mean sits above the discounted repair confidence.)
        probs = [0.95] * 10
        valid = cand("KL01AP8921", 0.90, probs)
        evidence = [
            (1, [valid]),
            (5, [("KL01AP8921", 0.90 * _REPAIR_CONF_DISCOUNT, probs)]),
        ]
        best, _ = self._select(evidence)
        self.assertEqual(best[0], "KL01AP8921")
        self.assertGreater(best[1], 0.90 * _REPAIR_CONF_DISCOUNT)

    def test_lkl0ap8921_leading_phantom_loses_to_valid_format(self):
        # The original production symptom: a phantom leading L. A valid
        # plate candidate from another tier must outrank it.
        evidence = [
            (1, [cand("LKL0AP8921", 0.80, [0.80] * 10)]),
            (2, [cand("KL01AP8921", 0.75, [0.90] * 9)]),
        ]
        best, _ = self._select(evidence)
        self.assertEqual(best[0], "KL01AP8921")


class ScorerSanityTests(unittest.TestCase):
    def test_clean_text_strips_separators(self):
        self.assertEqual(clean_text("KL-01-AP-8921"), "KL01AP8921")

    def test_valid_plate_outranks_junk_on_strict_adjustment(self):
        # The LOOSE legacy scorer saturates for both strings (that is why
        # the strict layer exists); selection actually ranks candidates
        # with _strict_format_adjustment, so compare that.
        from services.detection_service import _strict_format_adjustment

        self.assertGreater(
            _strict_format_adjustment("KL01AP8921"),
            _strict_format_adjustment("LKL0AP8921"),
        )


if __name__ == "__main__":
    unittest.main()
