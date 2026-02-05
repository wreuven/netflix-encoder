#!/usr/bin/env python3
"""
POC 2: Skip-Unchanged Encoding Handler

Encodes only changed/video_only frames, skipping unchanged frames entirely.
Uses VFR (variable frame rate) output so decoder holds last frame.

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler poc2_skip_unchanged_handler.py

Output:
    - output/skip_unchanged.mp4
    - output/skip_unchanged_metrics.json
"""

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
sys.path.insert(0, str(TRACER_PATH))

from frame_handler import BaseFrameHandler


class SkipUnchangedEncoder(BaseFrameHandler):
    """Encodes only changed frames, skipping unchanged ones."""

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_path = self.output_dir / "skip_unchanged.mp4"
        self.metrics_path = self.output_dir / "skip_unchanged_metrics.json"

        self.frame_count = 0
        self.frames_encoded = 0
        self.start_time = None
        self.category_counts = Counter()

        # Start FFmpeg with VFR input
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgra",
                "-s", f"{width}x{height}",
                "-use_wallclock_as_timestamps", "1",
                "-i", "pipe:",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-vsync", "vfr",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        print(f"[poc2] Skip-unchanged encoder started: {width}x{height}")
        print(f"[poc2] Output: {self.output_path}")

    def on_frame(self, evt, pixels, info):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1
        self.category_counts[evt.category] += 1

        # Skip unchanged frames
        if evt.category == "unchanged":
            return

        # Encode video_only and changed frames
        if pixels is not None:
            self.ffmpeg.stdin.write(pixels.tobytes())
            self.frames_encoded += 1

        # Status every 100 frames
        if self.frame_count % 100 == 0:
            elapsed = time.monotonic() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            skip_pct = 100 * (1 - self.frames_encoded / self.frame_count) if self.frame_count > 0 else 0
            print(f"[poc2] Frame {self.frame_count} ({fps:.1f} fps) - "
                  f"encoded: {self.frames_encoded}, skipped: {skip_pct:.1f}%")

    def on_phase_start(self, name):
        print(f"[poc2] Phase: {name}")

    def set_drop_count(self, count):
        pass

    def close(self):
        self.ffmpeg.stdin.close()
        stderr = self.ffmpeg.stderr.read().decode('utf-8', errors='replace')
        self.ffmpeg.wait()

        if self.ffmpeg.returncode != 0:
            print(f"[poc2] FFmpeg error:\n{stderr[-1000:]}")

        elapsed = time.monotonic() - self.start_time if self.start_time else 0
        output_size = self.output_path.stat().st_size if self.output_path.exists() else 0

        # Calculate metrics
        # Use frame_count for duration (what we would have had at constant rate)
        nominal_duration = self.frame_count / 30.0
        bitrate = (output_size * 8 / 1_000_000) / nominal_duration if nominal_duration > 0 else 0
        skip_pct = 100 * (self.frame_count - self.frames_encoded) / self.frame_count if self.frame_count > 0 else 0

        metrics = {
            "frames_total": self.frame_count,
            "frames_encoded": self.frames_encoded,
            "frames_skipped": self.frame_count - self.frames_encoded,
            "skip_percentage": round(skip_pct, 1),
            "frame_width": self.width,
            "frame_height": self.height,
            "nominal_duration_sec": round(nominal_duration, 2),
            "total_bytes": output_size,
            "avg_bitrate_mbps": round(bitrate, 3),
            "encode_time_sec": round(elapsed, 2),
            "category_counts": dict(self.category_counts),
            "output_file": str(self.output_path),
        }

        # Compare with baseline if available
        baseline_path = self.output_dir / "baseline_metrics.json"
        if baseline_path.exists():
            try:
                with open(baseline_path) as f:
                    baseline = json.load(f)
                baseline_bytes = baseline.get("total_bytes", 0)
                baseline_bitrate = baseline.get("avg_bitrate_mbps", 0)
                if baseline_bytes > 0:
                    metrics["baseline_comparison"] = {
                        "baseline_bytes": baseline_bytes,
                        "baseline_bitrate_mbps": baseline_bitrate,
                        "byte_savings_pct": round(100 * (1 - output_size / baseline_bytes), 1),
                        "bitrate_savings_pct": round(100 * (1 - bitrate / baseline_bitrate), 1) if baseline_bitrate > 0 else 0,
                    }
            except Exception as e:
                print(f"[poc2] Could not load baseline: {e}")

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print()
        print("=" * 60)
        print("POC 2: Skip-Unchanged Encoding Summary")
        print("=" * 60)
        print(f"  Total frames:     {self.frame_count}")
        print(f"  Frames encoded:   {self.frames_encoded}")
        print(f"  Frames skipped:   {self.frame_count - self.frames_encoded} ({skip_pct:.1f}%)")
        print(f"  Resolution:       {self.width}x{self.height}")
        print(f"  Duration:         {nominal_duration:.2f} sec")
        print(f"  Output size:      {output_size / 1_000_000:.2f} MB")
        print(f"  Bitrate:          {bitrate:.3f} Mbps")
        print()
        print("  Categories:")
        for cat, count in self.category_counts.items():
            pct = 100 * count / self.frame_count if self.frame_count > 0 else 0
            print(f"    {cat:12s}: {count:6d} ({pct:5.1f}%)")

        if "baseline_comparison" in metrics:
            comp = metrics["baseline_comparison"]
            print()
            print("  vs Baseline:")
            print(f"    Baseline:       {comp['baseline_bytes'] / 1_000_000:.2f} MB ({comp['baseline_bitrate_mbps']:.3f} Mbps)")
            print(f"    Byte savings:   {comp['byte_savings_pct']:.1f}%")
            print(f"    Bitrate savings:{comp['bitrate_savings_pct']:.1f}%")

        print()
        print(f"  Output: {self.output_path}")
        print()
        print("Playback:")
        print(f"  python3 ../h264-newpoc/splicer/staging_player.py {self.output_path}")


def create_handler(fw, fh, chrome_height):
    return SkipUnchangedEncoder(fw, fh, chrome_height)
