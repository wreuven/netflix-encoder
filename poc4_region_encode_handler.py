#!/usr/bin/env python3
"""
POC 4: Region-Only Encoding Handler

Encodes only the dirty regions (video_rect) to measure actual encoding cost savings.
Composites regions onto last frame for visual validation.

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler poc4_region_encode_handler.py

Output:
    - output/region_composite.mp4 — full frame with region composited
    - output/poc4_metrics.json — comparison metrics
"""

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
sys.path.insert(0, str(TRACER_PATH))

from frame_handler import BaseFrameHandler


class RegionEncodeHandler(BaseFrameHandler):
    """Encodes only dirty regions and composites onto last frame."""

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height
        self.frame_area = width * height

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.composite_path = self.output_dir / "region_composite.mp4"
        self.metrics_path = self.output_dir / "poc4_metrics.json"

        self.frame_count = 0
        self.frames_with_region = 0
        self.start_time = None
        self.category_counts = Counter()

        # Track last frame for compositing
        self.last_frame = None

        # Track region encoding stats
        self.total_region_pixels = 0
        self.total_full_pixels = 0

        # Start FFmpeg for composite output
        self.ffmpeg = None
        self.ffmpeg_failed = False
        self._start_ffmpeg()

        print(f"[poc4] Region encoder started: {width}x{height}")
        print(f"[poc4] Chrome height: {chrome_height}")
        print(f"[poc4] Output: {self.composite_path}")

    def _start_ffmpeg(self):
        """Start FFmpeg encoder process."""
        # libx264 requires even dimensions for yuv420p
        # Use scale filter to pad to even dimensions if needed
        scale_w = self.width if self.width % 2 == 0 else self.width + 1
        scale_h = self.height if self.height % 2 == 0 else self.height + 1

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-s", f"{self.width}x{self.height}",
            "-r", "30",
            "-i", "pipe:",
            "-vf", f"scale={scale_w}:{scale_h}:flags=neighbor",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(self.composite_path),
        ]

        self.ffmpeg = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_frame(self, frame_data):
        """Write frame to FFmpeg with error handling."""
        if self.ffmpeg_failed or self.ffmpeg is None:
            return False

        try:
            self.ffmpeg.stdin.write(frame_data.tobytes())
            return True
        except (BrokenPipeError, OSError) as e:
            if not self.ffmpeg_failed:
                self.ffmpeg_failed = True
                # Try to get error message
                try:
                    stderr = self.ffmpeg.stderr.read().decode('utf-8', errors='replace')
                    print(f"[poc4] FFmpeg error: {stderr[-1000:]}")
                except:
                    print(f"[poc4] FFmpeg pipe error: {e}")
            return False

    def _mb_align(self, x, y, w, h):
        """Expand region to 16-pixel MB alignment with 1 MB border."""
        mb_x = (x // 16) * 16
        mb_y = (y // 16) * 16
        mb_x2 = ((x + w + 15) // 16) * 16
        mb_y2 = ((y + h + 15) // 16) * 16

        # Add 1 MB (16px) border
        mb_x = max(0, mb_x - 16)
        mb_y = max(0, mb_y - 16)
        mb_x2 = min(self.width, mb_x2 + 16)
        mb_y2 = min(self.height, mb_y2 + 16)

        return (mb_x, mb_y, mb_x2 - mb_x, mb_y2 - mb_y)

    def on_frame(self, evt, pixels, info):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1
        self.category_counts[evt.category] += 1
        self.total_full_pixels += self.frame_area

        # Handle unchanged frames
        if evt.category == "unchanged":
            if self.last_frame is not None:
                self._write_frame(self.last_frame)
            return

        # No pixels available
        if pixels is None:
            if self.last_frame is not None:
                self._write_frame(self.last_frame)
            return

        # Handle video_only frames - composite region onto last frame
        if evt.category == "video_only" and evt.video_rect:
            vx, vy, vw, vh = evt.video_rect

            # IMPORTANT: video_rect is in CSS viewport coords (below chrome)
            # pixels array is in swapchain coords (includes chrome)
            vy = vy + self.chrome_height

            # Clamp to frame bounds
            vx = max(0, min(vx, self.width - 1))
            vy = max(0, min(vy, self.height - 1))
            vw = min(vw, self.width - vx)
            vh = min(vh, self.height - vy)

            if vw > 0 and vh > 0:
                # Get MB-aligned region
                ax, ay, aw, ah = self._mb_align(vx, vy, vw, vh)

                # Count region pixels
                self.total_region_pixels += aw * ah
                self.frames_with_region += 1

                # Composite: start with last frame, overlay new region
                if self.last_frame is not None:
                    composite = self.last_frame.copy()
                    # Copy only the changed region
                    composite[ay:ay+ah, ax:ax+aw] = pixels[ay:ay+ah, ax:ax+aw]
                else:
                    composite = pixels.copy()

                self._write_frame(composite)
                self.last_frame = composite
            else:
                # Invalid region
                self._write_frame(pixels)
                self.last_frame = pixels.copy()
        else:
            # Changed frame - encode full frame
            self.total_region_pixels += self.frame_area
            self._write_frame(pixels)
            self.last_frame = pixels.copy()

        # Status every 100 frames
        if self.frame_count % 100 == 0:
            elapsed = time.monotonic() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            region_pct = 100 * self.total_region_pixels / self.total_full_pixels if self.total_full_pixels > 0 else 0
            print(f"[poc4] Frame {self.frame_count} ({fps:.1f} fps) - "
                  f"region pixels: {region_pct:.1f}% of full")

    def on_phase_start(self, name):
        print(f"[poc4] Phase: {name}")

    def set_drop_count(self, count):
        pass

    def close(self):
        if self.ffmpeg is None:
            return

        try:
            self.ffmpeg.stdin.close()
        except:
            pass

        try:
            stderr = self.ffmpeg.stderr.read().decode('utf-8', errors='replace')
            self.ffmpeg.wait(timeout=10)
            if self.ffmpeg.returncode != 0 and stderr:
                print(f"[poc4] FFmpeg stderr: {stderr[-500:]}")
        except:
            pass

        elapsed = time.monotonic() - self.start_time if self.start_time else 0

        try:
            output_size = self.composite_path.stat().st_size
        except:
            output_size = 0

        duration = self.frame_count / 30.0
        composite_bitrate = (output_size * 8 / 1_000_000) / duration if duration > 0 else 0

        # Calculate theoretical savings
        region_ratio = self.total_region_pixels / self.total_full_pixels if self.total_full_pixels > 0 else 1
        theoretical_savings = 100 * (1 - region_ratio)

        # Load baseline for comparison
        baseline_path = self.output_dir / "baseline_metrics.json"
        baseline_comparison = {}
        if baseline_path.exists():
            try:
                with open(baseline_path) as f:
                    baseline = json.load(f)
                baseline_bytes = baseline.get("total_bytes", 0)
                baseline_bitrate = baseline.get("avg_bitrate_mbps", 0)

                estimated_region_bytes = int(baseline_bytes * region_ratio)
                estimated_region_bitrate = baseline_bitrate * region_ratio

                baseline_comparison = {
                    "baseline_bytes": baseline_bytes,
                    "baseline_bitrate_mbps": baseline_bitrate,
                    "estimated_region_bytes": estimated_region_bytes,
                    "estimated_region_bitrate_mbps": round(estimated_region_bitrate, 3),
                    "estimated_savings_pct": round(100 * (1 - region_ratio), 1),
                }
            except:
                pass

        metrics = {
            "frame_count": self.frame_count,
            "frames_with_region": self.frames_with_region,
            "frame_width": self.width,
            "frame_height": self.height,
            "chrome_height": self.chrome_height,
            "duration_sec": round(duration, 2),
            "total_full_pixels": self.total_full_pixels,
            "total_region_pixels": self.total_region_pixels,
            "region_pixel_ratio": round(region_ratio, 4),
            "theoretical_savings_pct": round(theoretical_savings, 1),
            "composite_bytes": output_size,
            "composite_bitrate_mbps": round(composite_bitrate, 3),
            "category_counts": dict(self.category_counts),
            "encode_time_sec": round(elapsed, 2),
        }

        if baseline_comparison:
            metrics["baseline_comparison"] = baseline_comparison

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print()
        print("=" * 60)
        print("POC 4: Region-Only Encoding Summary")
        print("=" * 60)
        print(f"  Total frames:       {self.frame_count}")
        print(f"  Frames with region: {self.frames_with_region}")
        print(f"  Resolution:         {self.width}x{self.height}")
        print(f"  Duration:           {duration:.2f} sec")
        print()
        print("  Pixel Count Analysis:")
        print(f"    Full frame pixels:   {self.total_full_pixels:,}")
        print(f"    Region-only pixels:  {self.total_region_pixels:,}")
        print(f"    Ratio:               {100*region_ratio:.1f}%")
        print(f"    Theoretical savings: {theoretical_savings:.1f}%")
        print()
        print("  Categories:")
        for cat, count in self.category_counts.items():
            pct = 100 * count / self.frame_count if self.frame_count > 0 else 0
            print(f"    {cat:12s}: {count:6d} ({pct:5.1f}%)")

        if baseline_comparison:
            print()
            print("  vs Baseline (estimated region-only):")
            print(f"    Baseline:            {baseline_comparison['baseline_bitrate_mbps']:.3f} Mbps")
            print(f"    Est. region-only:    {baseline_comparison['estimated_region_bitrate_mbps']:.3f} Mbps")
            print(f"    Est. savings:        {baseline_comparison['estimated_savings_pct']:.1f}%")

        print()
        print(f"  Composite output: {self.composite_path}")
        print(f"  Metrics:          {self.metrics_path}")
        print()
        print("Playback:")
        print(f"  python3 ../h264-newpoc/splicer/staging_player.py {self.composite_path}")


def create_handler(fw, fh, chrome_height):
    return RegionEncodeHandler(fw, fh, chrome_height)
