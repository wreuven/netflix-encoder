#!/usr/bin/env python3
"""
POC 3: Region Extraction Analysis Handler

Analyzes dirty regions (video_rect and damage_rects) to understand sizes,
positions, and potential savings from region-only encoding.

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler poc3_region_handler.py

Output:
    - output/regions/ — sample region crops as PNGs
    - output/region_heatmap.png — visualization of change locations
    - output/region_stats.json — size/position statistics
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
sys.path.insert(0, str(TRACER_PATH))

from frame_handler import BaseFrameHandler

# Optional PIL for images
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[poc3] PIL not installed - region crops won't be saved")


@dataclass
class RegionInfo:
    frame: int
    category: str  # "video" or "damage"
    x: int
    y: int
    width: int
    height: int
    area: int
    area_pct: float


class RegionAnalysisHandler(BaseFrameHandler):
    """Analyzes dirty regions without encoding."""

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height
        self.frame_area = width * height

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.regions_dir = self.output_dir / "regions"
        self.regions_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.start_time = None
        self.samples_saved = 0
        self.max_samples = 50

        self.video_regions: List[RegionInfo] = []
        self.damage_regions: List[RegionInfo] = []

        # Heatmap accumulator
        self.heatmap = np.zeros((height, width), dtype=np.float32)

        print(f"[poc3] Region analysis started: {width}x{height}")
        print(f"[poc3] Regions dir: {self.regions_dir}")

    def _mb_align(self, x, y, w, h, add_border=True):
        """Expand region to MB alignment with optional border."""
        mb_x = (x // 16) * 16
        mb_y = (y // 16) * 16
        mb_x2 = ((x + w + 15) // 16) * 16
        mb_y2 = ((y + h + 15) // 16) * 16

        if add_border:
            mb_x = max(0, mb_x - 16)
            mb_y = max(0, mb_y - 16)
            mb_x2 = min(self.width, mb_x2 + 16)
            mb_y2 = min(self.height, mb_y2 + 16)

        return (mb_x, mb_y, mb_x2 - mb_x, mb_y2 - mb_y)

    def on_frame(self, evt, pixels, info):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1

        # Process video_rect
        if evt.video_rect:
            vx, vy, vw, vh = evt.video_rect
            # Clamp to bounds
            vx = max(0, min(vx, self.width - 1))
            vy = max(0, min(vy, self.height - 1))
            vw = min(vw, self.width - vx)
            vh = min(vh, self.height - vy)

            if vw > 0 and vh > 0:
                area = vw * vh
                self.video_regions.append(RegionInfo(
                    frame=self.frame_count,
                    category="video",
                    x=vx, y=vy, width=vw, height=vh,
                    area=area,
                    area_pct=100 * area / self.frame_area,
                ))
                self.heatmap[vy:vy+vh, vx:vx+vw] += 1

                # Save sample
                if HAS_PIL and pixels is not None and self.samples_saved < self.max_samples:
                    self._save_crop(pixels, vx, vy, vw, vh, f"video_{self.frame_count:05d}.png")

        # Process damage_rects
        if evt.damage_rects:
            for rect in evt.damage_rects:
                dx, dy, dw, dh = rect.x, rect.y, rect.width, rect.height
                dx = max(0, min(dx, self.width - 1))
                dy = max(0, min(dy, self.height - 1))
                dw = min(dw, self.width - dx)
                dh = min(dh, self.height - dy)

                if dw > 0 and dh > 0:
                    area = dw * dh
                    self.damage_regions.append(RegionInfo(
                        frame=self.frame_count,
                        category="damage",
                        x=dx, y=dy, width=dw, height=dh,
                        area=area,
                        area_pct=100 * area / self.frame_area,
                    ))
                    self.heatmap[dy:dy+dh, dx:dx+dw] += 1

                    # Save sample (if no video rect already saved this frame)
                    if HAS_PIL and pixels is not None and self.samples_saved < self.max_samples:
                        if not evt.video_rect:
                            self._save_crop(pixels, dx, dy, dw, dh, f"damage_{self.frame_count:05d}.png")

        # Status every 200 frames
        if self.frame_count % 200 == 0:
            print(f"[poc3] Frame {self.frame_count} - "
                  f"video: {len(self.video_regions)}, damage: {len(self.damage_regions)}")

    def _save_crop(self, pixels, x, y, w, h, filename):
        """Save a region crop as PNG."""
        try:
            crop = pixels[y:y+h, x:x+w].copy()
            # BGRA -> RGBA
            crop = crop[:, :, [2, 1, 0, 3]]
            img = Image.fromarray(crop, mode="RGBA")
            img.save(self.regions_dir / filename)
            self.samples_saved += 1
        except Exception as e:
            pass  # Silently skip failed crops

    def on_phase_start(self, name):
        print(f"[poc3] Phase: {name}")

    def set_drop_count(self, count):
        pass

    def close(self):
        elapsed = time.monotonic() - self.start_time if self.start_time else 0

        # Save heatmap
        if HAS_PIL and self.heatmap.max() > 0:
            heatmap_norm = (255 * self.heatmap / self.heatmap.max()).astype(np.uint8)
            heatmap_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            heatmap_rgb[:, :, 0] = heatmap_norm  # Red channel
            img = Image.fromarray(heatmap_rgb, mode="RGB")
            img.save(self.output_dir / "region_heatmap.png")
            print(f"[poc3] Saved heatmap")

        # Compute statistics
        stats = self._compute_stats()

        with open(self.output_dir / "region_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print()
        print("=" * 60)
        print("POC 3: Region Extraction Analysis Summary")
        print("=" * 60)
        print(f"  Frame dimensions:   {self.width}x{self.height}")
        print(f"  Total frames:       {self.frame_count}")
        print(f"  Elapsed:            {elapsed:.2f} sec")
        print()

        if self.video_regions:
            vr = stats["video_regions"]
            print(f"  Video Regions ({vr['count']} total):")
            print(f"    Avg size:         {vr['avg_width']:.0f}x{vr['avg_height']:.0f}")
            print(f"    Avg area:         {vr['avg_area_pct']:.2f}% of frame")
            print(f"    MB-aligned area:  {vr['avg_aligned_area_pct']:.2f}% of frame")
            print()

        if self.damage_regions:
            dr = stats["damage_regions"]
            print(f"  Damage Regions ({dr['count']} total):")
            print(f"    Avg size:         {dr['avg_width']:.0f}x{dr['avg_height']:.0f}")
            print(f"    Avg area:         {dr['avg_area_pct']:.2f}% of frame")
            print(f"    MB-aligned area:  {dr['avg_aligned_area_pct']:.2f}% of frame")
            print()

        if "potential_savings" in stats:
            ps = stats["potential_savings"]
            print(f"  Potential Savings:")
            print(f"    Avg region:       {ps['avg_region_pct']:.2f}% of frame")
            print(f"    With MB align:    {ps['avg_aligned_pct']:.2f}% of frame")
            print(f"    Est. reduction:   ~{ps['estimated_reduction_pct']:.0f}% fewer pixels to encode")
            print()

        print(f"  Outputs:")
        print(f"    Stats:   {self.output_dir}/region_stats.json")
        print(f"    Heatmap: {self.output_dir}/region_heatmap.png")
        print(f"    Samples: {self.regions_dir}/ ({self.samples_saved} files)")

    def _compute_stats(self):
        stats = {
            "frame_dimensions": {"width": self.width, "height": self.height},
            "frame_count": self.frame_count,
            "video_regions": {"count": len(self.video_regions)},
            "damage_regions": {"count": len(self.damage_regions)},
        }

        if self.video_regions:
            areas = [r.area for r in self.video_regions]
            area_pcts = [r.area_pct for r in self.video_regions]
            widths = [r.width for r in self.video_regions]
            heights = [r.height for r in self.video_regions]

            aligned_areas = []
            for r in self.video_regions:
                _, _, aw, ah = self._mb_align(r.x, r.y, r.width, r.height)
                aligned_areas.append(aw * ah)

            stats["video_regions"].update({
                "avg_width": round(np.mean(widths), 1),
                "avg_height": round(np.mean(heights), 1),
                "avg_area": round(np.mean(areas), 0),
                "avg_area_pct": round(np.mean(area_pcts), 2),
                "avg_aligned_area": round(np.mean(aligned_areas), 0),
                "avg_aligned_area_pct": round(100 * np.mean(aligned_areas) / self.frame_area, 2),
            })

        if self.damage_regions:
            areas = [r.area for r in self.damage_regions]
            area_pcts = [r.area_pct for r in self.damage_regions]
            widths = [r.width for r in self.damage_regions]
            heights = [r.height for r in self.damage_regions]

            aligned_areas = []
            for r in self.damage_regions:
                _, _, aw, ah = self._mb_align(r.x, r.y, r.width, r.height)
                aligned_areas.append(aw * ah)

            stats["damage_regions"].update({
                "avg_width": round(np.mean(widths), 1),
                "avg_height": round(np.mean(heights), 1),
                "avg_area": round(np.mean(areas), 0),
                "avg_area_pct": round(np.mean(area_pcts), 2),
                "avg_aligned_area": round(np.mean(aligned_areas), 0),
                "avg_aligned_area_pct": round(100 * np.mean(aligned_areas) / self.frame_area, 2),
            })

        # Combined savings estimate
        all_regions = self.video_regions + self.damage_regions
        if all_regions:
            all_area_pcts = [r.area_pct for r in all_regions]
            all_aligned = []
            for r in all_regions:
                _, _, aw, ah = self._mb_align(r.x, r.y, r.width, r.height)
                all_aligned.append(aw * ah)

            avg_aligned_pct = 100 * np.mean(all_aligned) / self.frame_area
            stats["potential_savings"] = {
                "avg_region_pct": round(np.mean(all_area_pcts), 2),
                "avg_aligned_pct": round(avg_aligned_pct, 2),
                "estimated_reduction_pct": round(100 - avg_aligned_pct, 1),
            }

        return stats


def create_handler(fw, fh, chrome_height):
    return RegionAnalysisHandler(fw, fh, chrome_height)
