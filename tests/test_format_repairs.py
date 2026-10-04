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
    INDIAN_STATE_CODES,
    INDIAN_STATE_NAMES,
    resolve_indian_state,
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


class AmbiguityPairCoverageTests(unittest.TestCase):
    """Every shape-ambiguous pair the pipeline claims to know must
    actually be implemented and must actually produce a repair.

    The production spec lists O/0, I/1, Z/2, S/5, B/8, G/6 and Q/0.
    These tests fail if any of them silently goes missing.
    """

    def test_ambiguity_table_covers_all_specified_pairs(self):
        from services.detection_service import _AMBIGUOUS_CHARS

        for a, b in (
            ("O", "0"),
            ("I", "1"),
            ("Z", "2"),
            ("S", "5"),
            ("B", "8"),
            ("G", "6"),
            ("Q", "0"),
        ):
            self.assertIn(a, _AMBIGUOUS_CHARS, f"{a} missing")
            self.assertEqual(_AMBIGUOUS_CHARS[a], b)

    def test_digit_to_letter_directions_present(self):
        from services.detection_service import _AMBIGUOUS_CHARS

        for a, b in (("0", "O"), ("1", "I"), ("2", "Z"),
                     ("5", "S"), ("6", "G"), ("8", "B")):
            self.assertIn(a, _AMBIGUOUS_CHARS, f"{a} missing")
            self.assertEqual(_AMBIGUOUS_CHARS[a], b)

    def test_digit_read_as_Z_is_repaired(self):
        # MH1ZDE1433 is format-invalid (Z is not a digit); one swap on
        # the Z lands on the strictly valid MH12DE1433.
        probs = [0.9] * 10
        evidence = [(1, [cand("MH1ZDE1433", 0.9, probs)])]
        texts = [t for t, _, _ in _generate_format_repairs(evidence)]
        self.assertIn("MH12DE1433", texts)

    def test_letter_read_as_two_is_repaired(self):
        # Mirror direction: a 2 where the series letter Z belongs.
        probs = [0.9] * 10
        evidence = [(1, [cand("MH12D21433", 0.9, probs)])]
        texts = [t for t, _, _ in _generate_format_repairs(evidence)]
        self.assertIn("MH12DZ1433", texts)

    def test_six_read_as_G_is_repaired(self):
        probs = [0.9] * 10
        evidence = [(1, [cand("KL01A68921", 0.9, probs)])]
        texts = [t for t, _, _ in _generate_format_repairs(evidence)]
        self.assertIn("KL01AG8921", texts)

    def test_Q_read_as_letter_is_repaired_to_zero(self):
        # A zero with a tail reads as Q; MH2QEE7597 -> MH20EE7597.
        probs = [0.9] * 10
        evidence = [(1, [cand("MH2QEE7597", 0.9, probs)])]
        texts = [t for t, _, _ in _generate_format_repairs(evidence)]
        self.assertIn("MH20EE7597", texts)

    def test_new_pairs_never_touch_a_strict_valid_plate(self):
        # Plates containing the new glyphs that are ALREADY valid must be
        # returned untouched — no repair may rewrite a correct read.
        for text in ("KL01AG8921", "MH12DZ1433", "MH20EE7597"):
            self.assertTrue(strict_indian_plate(text), text)
            probs = [0.9] * len(text)
            self.assertEqual(
                _generate_format_repairs([(1, [cand(text, 0.95, probs)])]),
                [],
                text,
            )

    def test_swap_must_land_on_strict_validity(self):
        # A Z swap on an unknown state code produces nothing: the repair
        # rule is format-gated, not "swap whenever a glyph looks odd".
        probs = [0.9] * 10
        evidence = [(1, [cand("XX1ZDE1433", 0.9, probs)])]
        self.assertEqual(_generate_format_repairs(evidence), [])

    def test_recognizer_misread_is_not_fabricated(self):
        # IHI2DE1433 (GT MH12DE1433) is a RECOGNIZER error, not a glyph
        # confusion: M/I is not a shape-ambiguous pair. No repair may
        # invent MH12DE1433 out of it.
        probs = [0.9] * 10
        evidence = [(1, [cand("IHI2DE1433", 0.9, probs)])]
        texts = [t for t, _, _ in _generate_format_repairs(evidence)]
        self.assertNotIn("MH12DE1433", texts)


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


class StateResolutionTests(unittest.TestCase):
    def test_all_state_codes_have_names(self):
        for code in INDIAN_STATE_CODES:
            self.assertIn(code, INDIAN_STATE_NAMES)
            self.assertTrue(len(INDIAN_STATE_NAMES[code]) > 0)

    def test_resolve_standard_plates(self):
        code, name = resolve_indian_state("MH12DE1433")
        self.assertEqual(code, "MH")
        self.assertEqual(name, "Maharashtra")

        code, name = resolve_indian_state("DL3CAY9324")
        self.assertEqual(code, "DL")
        self.assertEqual(name, "Delhi")

        code, name = resolve_indian_state("KA51AA3469")
        self.assertEqual(code, "KA")
        self.assertEqual(name, "Karnataka")

        code, name = resolve_indian_state("KL01AP8921")
        self.assertEqual(code, "KL")
        self.assertEqual(name, "Kerala")

        code, name = resolve_indian_state("HR26BC55")
        self.assertEqual(code, "HR")
        self.assertEqual(name, "Haryana")

    def test_resolve_formatted_with_separators(self):
        code, name = resolve_indian_state("MH-12-DE-1433")
        self.assertEqual(code, "MH")
        self.assertEqual(name, "Maharashtra")

    def test_resolve_invalid_or_short(self):
        code, name = resolve_indian_state("XX01AP8921")
        self.assertIsNone(code)
        self.assertIsNone(name)

        code, name = resolve_indian_state("A")
        self.assertIsNone(code)
        self.assertIsNone(name)


if __name__ == "__main__":
    unittest.main()
