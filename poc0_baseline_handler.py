#!/usr/bin/env python3
"""
POC 0: Baseline Metrics Handler

Encodes all frames (unchanged, video_only, changed) to establish baseline.

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler poc0_baseline_handler.py

Output:
    - output/baseline.mp4
    - output/baseline_metrics.json
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Add chrome_gpu_tracer to path
TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
sys.path.insert(0, str(TRACER_PATH))

from frame_handler import BaseFrameHandler


class BaselineEncoder(BaseFrameHandler):
    """Encodes all frames to MP4 for baseline metrics."""

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_path = self.output_dir / "baseline.mp4"
        self.metrics_path = self.output_dir / "baseline_metrics.json"

        self.frame_count = 0
        self.start_time = None
        self.category_counts = {"unchanged": 0, "video_only": 0, "changed": 0}

        # Start FFmpeg
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgra",
                "-s", f"{width}x{height}",
                "-r", "30",
                "-i", "pipe:",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        print(f"[poc0] Baseline encoder started: {width}x{height}")
        print(f"[poc0] Output: {self.output_path}")

    def on_frame(self, evt, pixels, info):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1
        self.category_counts[evt.category] += 1

        # Write every frame (including unchanged - use last pixels)
        if pixels is not None:
            self.last_pixels = pixels
            self.ffmpeg.stdin.write(pixels.tobytes())
        elif hasattr(self, 'last_pixels'):
            # For unchanged frames, repeat last frame
            self.ffmpeg.stdin.write(self.last_pixels.tobytes())

        # Status every 100 frames
        if self.frame_count % 100 == 0:
            elapsed = time.monotonic() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[poc0] Frame {self.frame_count} ({fps:.1f} fps)")

    def on_phase_start(self, name):
        print(f"[poc0] Phase: {name}")

    def set_drop_count(self, count):
        if count > 0:
            print(f"[poc0] Dropped frames: {count}")

    def close(self):
        # Close FFmpeg
        self.ffmpeg.stdin.close()
        stderr = self.ffmpeg.stderr.read().decode('utf-8', errors='replace')
        self.ffmpeg.wait()

        if self.ffmpeg.returncode != 0:
            print(f"[poc0] FFmpeg error:\n{stderr[-1000:]}")

        elapsed = time.monotonic() - self.start_time if self.start_time else 0
        output_size = self.output_path.stat().st_size if self.output_path.exists() else 0
        duration = self.frame_count / 30.0
        bitrate = (output_size * 8 / 1_000_000) / duration if duration > 0 else 0

        metrics = {
            "frame_count": self.frame_count,
            "frame_width": self.width,
            "frame_height": self.height,
            "duration_sec": round(duration, 2),
            "total_bytes": output_size,
            "avg_bitrate_mbps": round(bitrate, 3),
            "encode_time_sec": round(elapsed, 2),
            "category_counts": self.category_counts,
            "output_file": str(self.output_path),
        }

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print()
        print("=" * 60)
        print("POC 0: Baseline Metrics Summary")
        print("=" * 60)
        print(f"  Frames:       {self.frame_count}")
        print(f"  Resolution:   {self.width}x{self.height}")
        print(f"  Duration:     {duration:.2f} sec")
        print(f"  Output size:  {output_size / 1_000_000:.2f} MB")
        print(f"  Bitrate:      {bitrate:.3f} Mbps")
        print(f"  Categories:   {self.category_counts}")
        print(f"  Output:       {self.output_path}")
        print()
        print("Playback:")
        print(f"  python3 ../h264-newpoc/splicer/staging_player.py {self.output_path}")


def create_handler(fw, fh, chrome_height):
    """Factory function called by test_frame_classifier.py"""
    return BaselineEncoder(fw, fh, chrome_height)
