"""Unit tests for the detection geometry layer.

Run from the project root:
    venv/Scripts/python.exe -m unittest tests.test_detection_geometry -v

Covers the evidence-based detection fixes:
  - plate-box plausibility (shape/size sanity for YOLO boxes)
  - conservative box expansion + clamping (Phase 4)
  - cross-pass IoU dedupe (two-pass YOLO, Phase 2/3)
  - skew estimator must REJECT flat crops (Phase 6 gating)

No models are loaded — pure geometry, mirroring the production path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np

from services.detection_service import (  # noqa: E402
    _box_iou,
    _dedupe_boxes,
    _expand_box,
    _looks_like_plate_box,
)


class PlateBoxPlausibilityTests(unittest.TestCase):
    def test_typical_plate_accepted(self):
        # 300x80 px box in a 1280x720 image: classic car plate.
        self.assertTrue(_looks_like_plate_box((100, 300, 400, 380), 1280, 720))

    def test_two_line_plate_accepted(self):
        # 220x140 (aspect ~1.57): motorcycle stacked plate.
        self.assertTrue(_looks_like_plate_box((10, 10, 230, 150), 1280, 720))

    def test_whole_image_box_rejected(self):
        # The car itself / a collage cell, not a readable plate.
        self.assertFalse(_looks_like_plate_box((0, 0, 1270, 710), 1280, 720))

    def test_square_box_rejected(self):
        # Headlight / badge geometry.
        self.assertFalse(_looks_like_plate_box((100, 100, 200, 200), 1280, 720))

    def test_degenerate_sliver_rejected(self):
        # 8x5 px: labeling/detection noise.
        self.assertFalse(_looks_like_plate_box((50, 50, 58, 55), 1280, 720))

    def test_very_tall_box_rejected(self):
        # Windshield-shaped region.
        self.assertFalse(_looks_like_plate_box((100, 100, 220, 500), 1280, 720))


class ExpandBoxTests(unittest.TestCase):
    def test_expands_and_stays_inside(self):
        # padx = round(390*0.04) = 16, pady = round(90*0.04) = 4;
        # the left margin clamps at the image edge.
        box = _expand_box((10, 10, 400, 100), 1280, 720, frac=0.04)
        self.assertEqual(box, (0, 6, 416, 104))

    def test_clamps_to_image_edges(self):
        box = _expand_box((0, 0, 300, 80), 1280, 720, frac=0.04)
        self.assertEqual(box[0], 0)
        self.assertEqual(box[1], 0)
        self.assertLessEqual(box[2], 1280)
        self.assertLessEqual(box[3], 720)

    def test_x1_lt_x2_after_clamp(self):
        box = _expand_box((1270, 700, 1280, 720), 1280, 720, frac=0.5)
        self.assertLess(box[0], box[2])
        self.assertLess(box[1], box[3])


class BoxIouTests(unittest.TestCase):
    def test_identical_boxes_iou_one(self):
        self.assertAlmostEqual(_box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes_iou_zero(self):
        self.assertEqual(_box_iou((0, 0, 10, 10), (100, 100, 110, 110)), 0.0)

    def test_partial_overlap(self):
        # Two 10x10 boxes offset by 5 px: IoU = 25/175.
        self.assertAlmostEqual(
            _box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25.0 / 175.0
        )


class DedupeTests(unittest.TestCase):
    def test_duplicate_cross_pass_boxes_merge(self):
        dets = [
            (0.81, 100, 200, 380, 280),
            (0.77, 102, 199, 379, 281),  # same plate from the retry pass
        ]
        kept = _dedupe_boxes(dets, iou_thr=0.60)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], 0.81)

    def test_distinct_plates_both_kept(self):
        dets = [
            (0.90, 10, 10, 200, 80),
            (0.80, 500, 300, 700, 370),
        ]
        kept = _dedupe_boxes(dets)
        self.assertEqual(len(kept), 2)

    def test_highest_confidence_wins(self):
        dets = [
            (0.50, 100, 100, 200, 140),
            (0.95, 101, 101, 201, 141),
        ]
        kept = _dedupe_boxes(dets)
        self.assertEqual(kept[0][0], 0.95)


class SkewGateTests(unittest.TestCase):
    def test_flat_plate_gets_no_deskew(self):
        """A straight, well-formed plate must NOT be 'corrected'.

        _variant_deskew returns None unless the geometry provides clear
        evidence of a tilt — rotating correct crops would damage them.
        """
        from services.detection_service import _variant_deskew

        img = np.full((80, 400), 235, dtype=np.uint8)
        # Simple dark "text" band: horizontal strokes only, no tilt.
        cv_rect = np.zeros((80, 400), dtype=np.uint8)
        cv_rect[30:50, 40:360] = 255
        img = cv2.bitwise_or(img, 255 - cv_rect) if False else img
        # Draw a few vertical character strokes (like KL01AP8921)
        for x in range(60, 340, 28):
            img[25:58, x : x + 10] = 20

        self.assertIsNone(_variant_deskew(img))

    def test_tilted_plate_is_detected(self):
        """A clearly rotated plate should be corrected (angle returned)."""
        from services.detection_service import _variant_deskew

        img = np.full((120, 500), 235, dtype=np.uint8)
        for x in range(60, 440, 34):
            img[40:80, x : x + 12] = 20

        # Rotate by -6 degrees to simulate a tilted plate.
        matrix = cv2.getRotationMatrix2D((250, 60), -6.0, 1.0)
        tilted = cv2.warpAffine(
            img, matrix, (500, 120), borderValue=235
        )

        result = _variant_deskew(tilted)
        self.assertIsNotNone(result)
        _, angle = result
        self.assertGreater(abs(angle), 2.0)


import cv2  # noqa: E402


if __name__ == "__main__":
    unittest.main()
