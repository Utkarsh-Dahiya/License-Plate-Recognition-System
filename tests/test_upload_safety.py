"""
Upload-safety regression tests for the video staging path (Phase 5).

Covers the hardening of POST /api/process/video: the temp file an upload is
staged through now takes its suffix from an allowlist of known video
containers instead of copying it out of the client-supplied filename.

Importing `main` loads FastAPI and the data loader, but NOT the ML models —
those load lazily on the first detection request — so these tests stay fast and
never touch the weights.

Run from the project root:
    venv/Scripts/python.exe -m unittest tests.test_upload_safety -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from main import _ALLOWED_VIDEO_SUFFIXES, _safe_video_suffix  # noqa: E402


class SafeVideoSuffixTests(unittest.TestCase):
    """The staged temp file must never be steered outside the temp directory."""

    def test_ordinary_names_keep_their_extension(self):
        self.assertEqual(_safe_video_suffix("clip.mp4"), ".mp4")
        self.assertEqual(_safe_video_suffix("holiday.MOV"), ".mov")
        self.assertEqual(_safe_video_suffix("a.b.mkv"), ".mkv")

    def test_every_allowed_suffix_round_trips(self):
        for ext in _ALLOWED_VIDEO_SUFFIXES:
            with self.subTest(ext=ext):
                self.assertEqual(_safe_video_suffix(f"clip{ext}"), ext)

    def test_path_separators_cannot_reach_the_suffix(self):
        # The bug this guards: a name like "clip.mp4/nonexistentdir" produced
        # the suffix "./nonexistentdir", which tempfile turned into a path
        # OUTSIDE the temp directory — or raised FileNotFoundError, surfacing
        # to the client as an unhandled HTTP 500.
        for name in (
            "clip.mp4/nonexistentdir",
            "clip.mp4/../../evil",
            "..\\..\\evil.mp4",
            "clip.mp4/../lva_esc/pwned",
            "/etc/passwd.mp4",
        ):
            with self.subTest(name=name):
                self.assertEqual(_safe_video_suffix(name), ".mp4")

    def test_unknown_or_missing_extension_falls_back(self):
        for name in ("noext", "archive.tar.gz", "payload.exe", "", None):
            with self.subTest(name=name):
                self.assertEqual(_safe_video_suffix(name), ".mp4")

    def test_suffix_is_always_a_bare_allowlisted_extension(self):
        # The invariant the caller depends on, for ANY input: a plain
        # extension, no separators, no dot-dot.
        hostile = [
            "../../../etc/passwd",
            "x" * 5000 + ".mp4",
            "....//....//x.mp4",
            "clip.mp4\x00.txt",
            ".mp4",
            "..",
            ".",
            "\u202e.mp4",
            "....mp4",
        ]
        for name in hostile:
            with self.subTest(name=name[:40]):
                suffix = _safe_video_suffix(name)
                self.assertIn(suffix, _ALLOWED_VIDEO_SUFFIXES)
                self.assertNotIn("/", suffix)
                self.assertNotIn("\\", suffix)
                self.assertNotIn("..", suffix)

    def test_generated_suffix_cannot_escape_the_temp_directory(self):
        """End to end: the path tempfile builds stays inside the temp dir."""
        for name in (
            "clip.mp4/nonexistentdir",
            "../../evil.mp4",
            "clip.exe",
            "normal.mp4",
        ):
            with self.subTest(name=name):
                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=_safe_video_suffix(name),
                )
                try:
                    self.assertEqual(
                        os.path.dirname(os.path.abspath(tmp.name)).lower(),
                        os.path.abspath(tempfile.gettempdir()).lower(),
                    )
                finally:
                    tmp.close()
                    os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()