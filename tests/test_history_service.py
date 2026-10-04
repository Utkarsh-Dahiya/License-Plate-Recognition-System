"""Unit tests for history_log search / stats / export (Phase 3-5).

Runs against a temp JSONL file patched over the real HISTORY_LOG_PATH,
so the real data file is never touched.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "app" / "backend"))

from services import history_log


class HistoryServiceTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "detections.jsonl"
        self._patcher = mock.patch.object(
            history_log, "HISTORY_LOG_PATH", self.log_path
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def _add_image(self, filename, text, confidence, status, t, latency=1.0):
        return history_log.log_entry(
            "image",
            filename=filename,
            plates_detected=1 if text else 0,
            best_plate_text=text,
            best_confidence=confidence,
            status=status,
            processing_time_seconds=latency,
            timestamp=t,
        )

    def _add_video(self, filename, t):
        return history_log.log_entry(
            "video",
            filename=filename,
            processed_frames=10,
            total_frames=10,
            detections_found=2,
            status="COMPLETED",
            processing_time_seconds=30.0,
            timestamp=t,
        )


class ReadHistoryTests(HistoryServiceTestBase):
    def test_roundtrip_and_pagination(self):
        self._add_image("a.jpg", "MH12AB1234", 0.9, "HIGH_CONFIDENCE", 100.0)
        self._add_image("b.jpg", "", 0.0, "NO_PLATE", 200.0)
        self._add_video("c.mp4", 300.0)

        out = history_log.read_history(page_size=2, page=1)
        self.assertEqual(out["total"], 3)
        self.assertEqual(len(out["results"]), 2)
        # Newest first.
        self.assertEqual(out["results"][0]["filename"], "c.mp4")

        out2 = history_log.read_history(page_size=2, page=2)
        self.assertEqual(len(out2["results"]), 1)
        self.assertEqual(out2["results"][0]["filename"], "a.jpg")

    def test_type_filter(self):
        self._add_image("a.jpg", "X", 0.5, "REVIEW", 1.0)
        self._add_video("c.mp4", 2.0)
        out = history_log.read_history(entry_type="video")
        self.assertEqual(out["total"], 1)
        self.assertEqual(out["results"][0]["type"], "video")

    def test_search_matches_filename_and_plate(self):
        self._add_image("taxi_fleet.jpg", "MH12AB1234", 0.9, "HIGH_CONFIDENCE", 1.0)
        self._add_image("bus.jpg", "KA01MX7788", 0.7, "REVIEW", 2.0)
        self._add_image("night_shot.png", "", 0.0, "NO_PLATE", 3.0)

        # By plate text, case-insensitive.
        out = history_log.read_history(search="ka01")
        self.assertEqual(out["total"], 1)
        self.assertEqual(out["results"][0]["filename"], "bus.jpg")

        # By filename.
        out = history_log.read_history(search="TAXI")
        self.assertEqual(out["total"], 1)
        self.assertEqual(out["results"][0]["best_plate_text"], "MH12AB1234")

        # No match -> empty, not error.
        out = history_log.read_history(search="zzz")
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["results"], [])


class HistoryStatsTests(HistoryServiceTestBase):
    def test_zero_state_when_no_history(self):
        stats = history_log.history_stats()
        self.assertEqual(stats["total_runs"], 0)
        self.assertIsNone(stats["ocr_text_rate"])
        self.assertIsNone(stats["avg_best_confidence"])
        self.assertIsNone(stats["latency"]["p95_seconds"])
        self.assertEqual(stats["status_distribution"], {})

    def test_stats_from_real_entries(self):
        self._add_image("a.jpg", "MH12AB1234", 0.90, "HIGH_CONFIDENCE", 1.0, latency=1.0)
        self._add_image("b.jpg", "KA01MX7788", 0.60, "REVIEW", 2.0, latency=2.0)
        self._add_image("c.jpg", "", 0.0, "NO_PLATE", 3.0, latency=3.0)
        self._add_video("v.mp4", 4.0)

        stats = history_log.history_stats()
        self.assertEqual(stats["total_runs"], 4)
        self.assertEqual(stats["image_runs"], 3)
        self.assertEqual(stats["video_runs"], 1)
        self.assertEqual(stats["plates_detected"], 2)
        self.assertEqual(stats["image_runs_with_text"], 2)
        self.assertAlmostEqual(stats["ocr_text_rate"], round(2 / 3, 4))
        self.assertAlmostEqual(stats["avg_best_confidence"], (0.90 + 0.60) / 2, places=3)
        self.assertAlmostEqual(stats["latency"]["mean_seconds"], 2.0, places=3)
        # Image-path latency only; the 30s video job is a different
        # workload class and is deliberately excluded.
        self.assertEqual(stats["latency"]["samples"], 3)
        self.assertEqual(stats["status_distribution"]["REVIEW"], 1)
        self.assertEqual(stats["status_distribution"]["NO_PLATE"], 1)
        self.assertEqual(stats["first_run_timestamp"], 1.0)
        self.assertEqual(stats["last_run_timestamp"], 4.0)


class ExportTests(HistoryServiceTestBase):
    def test_csv_export_fields_and_escaping(self):
        self._add_image('weird,"name".jpg', "MH12AB1234", 0.9, "HIGH_CONFIDENCE", 1.0)

        content, media = history_log.export_entries("csv")
        self.assertEqual(media, "text/csv")

        rows = list(csv.reader(io.StringIO(content)))
        self.assertEqual(rows[0], history_log.EXPORT_FIELDS)
        self.assertEqual(len(rows), 2)
        self.assertIn("MH12AB1234", rows[1])
        # The comma inside the filename must be quoted, not split.
        self.assertIn('weird,"name".jpg', rows[1])

    def test_json_export_structure(self):
        self._add_image("a.jpg", "MH12AB1234", 0.9, "HIGH_CONFIDENCE", 1.0)
        self._add_video("v.mp4", 2.0)

        content, media = history_log.export_entries("json", entry_type="image")
        self.assertEqual(media, "application/json")

        payload = json.loads(content)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["entries"][0]["best_plate_text"], "MH12AB1234")

    def test_export_respects_search(self):
        self._add_image("taxi.jpg", "MH12AB1234", 0.9, "HIGH_CONFIDENCE", 1.0)
        self._add_image("bus.jpg", "KA01MX7788", 0.7, "REVIEW", 2.0)

        content, _ = history_log.export_entries("csv", search="taxi")
        rows = list(csv.reader(io.StringIO(content)))
        self.assertEqual(len(rows), 2)  # header + 1 row
        self.assertIn("taxi.jpg", rows[1])


if __name__ == "__main__":
    unittest.main()
