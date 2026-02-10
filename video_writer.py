"""
Verification video writer: generates a color-coded MP4 showing every
captured browser frame with classification overlays.

Visual design per frame:
  - Browser pixels scaled to output resolution (default 960x540)
  - Color-coded 4px border: green=VIDEO_ONLY, red=CHANGED
  - Green bounding box on VIDEO_ONLY frames showing video element rect
  - HUD text (bottom-left): frame number, category, phase name

Uses PyAV for variable frame rate (VFR) output — only frames with actual
compositor output are written (UNCHANGED poll timeouts are skipped), each
stamped with its real wall-clock time.

The Vulkan capture layer is single-buffered: it only writes a new frame
after the reader acks the previous one.  This means no frames can be
lost — Chrome may present at 25Hz vsync, but the layer simply skips
writing on presents where the reader hasn't caught up yet.  Chrome also
throttles presents when nothing is changing (~2fps idle).

Requires: Pillow, PyAV (av).
"""

import fractions
import time
from typing import Optional

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category -> border color
_BORDER_COLORS = {
    "unchanged": (128, 128, 128),   # gray
    "video_only": (0, 200, 0),      # green
    "changed": (220, 40, 40),       # red
}

_BORDER_WIDTH = 4


class VerificationVideoWriter:
    """Write a color-coded verification MP4 via PyAV with VFR timestamps."""

    def __init__(self, output_path: str, source_width: int, source_height: int,
                 width: int = 960, height: int = 540, fps: int = 25,
                 chrome_height: int = 0):
        self.output_path = output_path
        self.source_width = source_width
        self.source_height = source_height
        self.width = width
        self.height = height
        self.fps = fps

        self._container = None
        self._stream = None
        self._phase_name: str = ""
        self._frame_number: int = 0
        self._start_time: float = 0.0
        self._drop_count: int = 0

        # Full-frame buffer at source resolution (RGB numpy array).
        # Tiles are blitted into this, then it's scaled for output.
        self._canvas = np.zeros((source_height, source_width, 3), dtype=np.uint8)

        # Browser chrome height — crop this many rows from the top of the
        # canvas so the output shows only the content area.
        self._chrome_height = chrome_height

        # Content area dimensions (source pixels after cropping chrome)
        self._content_height = source_height - chrome_height

        # Scale factors: map CSS viewport coords to output coords
        self._sx = width / source_width if source_width else 1.0
        self._sy = height / self._content_height if self._content_height else 1.0

        # Try to load a monospace font for the HUD
        self._font = None
        try:
            self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except (OSError, IOError):
            try:
                self._font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 16)
            except (OSError, IOError):
                self._font = ImageFont.load_default()

    def open(self):
        """Open the output container and configure the H.264 stream."""
        self._container = av.open(self.output_path, mode='w')
        # rate must match time_base denominator — PyAV/libx264 uses rate
        # to derive the codec time_base, so rate=1000 with time_base=1/1000
        # gives us millisecond PTS precision for VFR output.
        self._stream = self._container.add_stream('libx264', rate=1000)
        self._stream.width = self.width
        self._stream.height = self.height
        self._stream.pix_fmt = 'yuv420p'
        self._stream.options = {'preset': 'fast', 'crf': '23'}
        self._stream.time_base = fractions.Fraction(1, 1000)
        self._frame_number = 0
        self._start_time = 0.0

    def set_phase(self, name: str):
        """Set the current phase label shown in the HUD."""
        self._phase_name = name

    def set_drop_count(self, count: int):
        """Set the cumulative drop count shown in the HUD."""
        self._drop_count = count

    def write_frame(self, evt, pixels: Optional[np.ndarray], info=None):
        """Overlay classification info onto the frame and encode with VFR PTS.

        Args:
            evt: FrameEvent with .category and optional .video_rect
            pixels: BGRA numpy array (H, W, 4) or None for UNCHANGED frames
            info: FrameInfo with .x, .y tile position, or None
        """
        if self._container is None:
            return

        now = time.monotonic()
        self._frame_number += 1
        if self._frame_number == 1:
            self._start_time = now

        # Skip UNCHANGED frames — no new compositor output.  VFR holds
        # the previous frame on screen until the next real frame arrives.
        if evt.category == "unchanged":
            return

        # Blit tile into full-frame canvas, then scale to output size
        if pixels is not None:
            # BGRA -> RGB
            rgb = pixels[:, :, [2, 1, 0]]
            th, tw = rgb.shape[:2]
            tx = info.x if info else 0
            ty = info.y if info else 0
            # Clamp to canvas bounds
            cx = max(0, tx)
            cy = max(0, ty)
            cw = min(tw, self.source_width - cx)
            ch = min(th, self.source_height - cy)
            if cw > 0 and ch > 0:
                self._canvas[cy:cy+ch, cx:cx+cw] = rgb[:ch, :cw]

        # Crop out browser chrome, then scale to output size
        content = self._canvas[self._chrome_height:, :]
        img = Image.fromarray(content).resize(
            (self.width, self.height), Image.BILINEAR)

        draw = ImageDraw.Draw(img)

        # Color-coded border
        color = _BORDER_COLORS.get(evt.category, (128, 128, 128))
        bw = _BORDER_WIDTH
        w, h = self.width, self.height
        draw.rectangle([0, 0, w - 1, bw - 1], fill=color)           # top
        draw.rectangle([0, h - bw, w - 1, h - 1], fill=color)       # bottom
        draw.rectangle([0, 0, bw - 1, h - 1], fill=color)           # left
        draw.rectangle([w - bw, 0, w - 1, h - 1], fill=color)       # right

        # Green rect for video_rect on VIDEO_ONLY frames
        if evt.category == "video_only" and evt.video_rect is not None:
            vx, vy, vw, vh = evt.video_rect
            # CSS viewport coords map directly after chrome crop
            rx0 = int(vx * self._sx)
            ry0 = int(vy * self._sy)
            rx1 = int((vx + vw) * self._sx)
            ry1 = int((vy + vh) * self._sy)
            # Clamp to output bounds
            rx0 = max(0, min(rx0, self.width - 1))
            ry0 = max(0, min(ry0, self.height - 1))
            rx1 = max(0, min(rx1, self.width - 1))
            ry1 = max(0, min(ry1, self.height - 1))
            # Only draw if rect is valid (at least 6px to fit 3 nested lines)
            if rx1 - rx0 >= 6 and ry1 - ry0 >= 6:
                for i in range(3):
                    draw.rectangle(
                        [rx0 + i, ry0 + i, rx1 - i, ry1 - i],
                        outline=(0, 255, 0),
                    )

        # HUD: frame number + category + rect + phase name + drop count
        # UNCHANGED frames are skipped (VFR), so only VIDEO_ONLY and
        # CHANGED appear.
        parts = [f"#{self._frame_number}"]

        if evt.category == "video_only":
            parts.append("VIDEO_ONLY")
            if evt.video_rect:
                vx, vy, vw, vh = evt.video_rect
                parts.append(f"video:({vx},{vy},{vw}x{vh})")
        elif evt.category == "changed":
            parts.append("CHANGED")
            if evt.damage_rects and len(evt.damage_rects) > 0:
                dr = evt.damage_rects[0]
                parts.append(f"damage:({dr.x},{dr.y},{dr.width}x{dr.height})")

        parts.append(self._phase_name)
        parts.append(f"DROP:{self._drop_count}")
        hud_text = "  ".join(parts)
        bbox = draw.textbbox((0, 0), hud_text, font=self._font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 6
        hud_x = pad
        hud_y = self.height - th - pad * 3
        draw.rectangle(
            [hud_x - pad, hud_y - pad, hud_x + tw + pad, hud_y + th + pad],
            fill=(0, 0, 0),
        )
        draw.text((hud_x, hud_y), hud_text, fill=(255, 255, 255), font=self._font)

        # Encode with wall-clock PTS (milliseconds)
        frame = av.VideoFrame.from_image(img)
        frame.pts = int((now - self._start_time) * 1000)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self):
        """Flush encoder and close the container."""
        if self._container is not None:
            # Flush buffered frames
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None
