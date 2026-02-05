#!/usr/bin/env python3
"""
POC 1: Frame Classification Logging Handler

Logs all frame classifications to understand the distribution of
unchanged/video_only/changed frames.

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler poc1_classifier_handler.py

Output:
    - output/classification_log.jsonl — per-frame data
    - output/classification_summary.json — statistics
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
sys.path.insert(0, str(TRACER_PATH))

from frame_handler import BaseFrameHandler


class ClassifierHandler(BaseFrameHandler):
    """Logs frame classifications for analysis."""

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / "classification_log.jsonl"
        self.summary_path = self.output_dir / "classification_summary.json"

        self.log_file = open(self.log_path, "w")
        self.frame_count = 0
        self.start_time = None
        self.counts = Counter()
        self.current_phase = ""

        # Sample storage
        self.video_rect_samples = []
        self.damage_rect_samples = []

        print(f"[poc1] Classification logger started: {width}x{height}")
        print(f"[poc1] Log: {self.log_path}")

    def on_frame(self, evt, pixels, info):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1
        self.counts[evt.category] += 1

        # Log frame data
        entry = {
            "frame": self.frame_count,
            "timestamp": evt.timestamp,
            "category": evt.category,
            "phase": self.current_phase,
            "video_rect": evt.video_rect,
            "damage_rects": [
                {"x": r.x, "y": r.y, "w": r.width, "h": r.height}
                for r in (evt.damage_rects or [])
            ] if evt.damage_rects else [],
        }
        self.log_file.write(json.dumps(entry) + "\n")

        # Collect samples
        if evt.video_rect and len(self.video_rect_samples) < 20:
            self.video_rect_samples.append({
                "frame": self.frame_count,
                "rect": evt.video_rect,
                "phase": self.current_phase,
            })

        if evt.damage_rects and len(self.damage_rect_samples) < 20:
            self.damage_rect_samples.append({
                "frame": self.frame_count,
                "rects": [{"x": r.x, "y": r.y, "w": r.width, "h": r.height}
                          for r in evt.damage_rects],
                "phase": self.current_phase,
            })

        # Status every 200 frames
        if self.frame_count % 200 == 0:
            elapsed = time.monotonic() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[poc1] Frame {self.frame_count} ({fps:.1f} fps) - "
                  f"U:{self.counts['unchanged']} V:{self.counts['video_only']} C:{self.counts['changed']}")

    def on_phase_start(self, name):
        self.current_phase = name
        print(f"[poc1] Phase: {name}")

    def set_drop_count(self, count):
        pass  # Ignore for logging

    def close(self):
        self.log_file.close()

        elapsed = time.monotonic() - self.start_time if self.start_time else 0
        total = sum(self.counts.values())

        summary = {
            "frame_count": self.frame_count,
            "frame_width": self.width,
            "frame_height": self.height,
            "chrome_height": self.chrome_height,
            "elapsed_sec": round(elapsed, 2),
            "fps": round(total / elapsed, 2) if elapsed > 0 else 0,
            "counts": dict(self.counts),
            "percentages": {
                k: round(100 * v / total, 1) if total > 0 else 0
                for k, v in self.counts.items()
            },
            "video_rect_samples": self.video_rect_samples,
            "damage_rect_samples": self.damage_rect_samples,
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print()
        print("=" * 60)
        print("POC 1: Frame Classification Summary")
        print("=" * 60)
        print(f"  Frames:       {self.frame_count}")
        print(f"  Duration:     {elapsed:.2f} sec")
        print(f"  Resolution:   {self.width}x{self.height}")
        print()
        print("  Category breakdown:")
        for cat in ["unchanged", "video_only", "changed"]:
            count = self.counts.get(cat, 0)
            pct = summary['percentages'].get(cat, 0)
            bar = "#" * int(pct / 2)
            print(f"    {cat:12s}: {count:6d} ({pct:5.1f}%) {bar}")
        print()
        print(f"  Log file:     {self.log_path}")
        print(f"  Summary:      {self.summary_path}")

        # Insights
        unchanged_pct = summary['percentages'].get('unchanged', 0)
        video_pct = summary['percentages'].get('video_only', 0)

        print()
        if unchanged_pct > 30:
            print(f"  Insight: {unchanged_pct:.1f}% unchanged - significant skip potential")
        if video_pct > 30:
            print(f"  Insight: {video_pct:.1f}% video_only - region encoding will help")


def create_handler(fw, fh, chrome_height):
    return ClassifierHandler(fw, fh, chrome_height)
