"""
Frame handler protocol for pluggable frame processing.

Handlers receive classified frame events and can process them however needed:
- Write to video file (default: VerificationVideoWriter)
- Stream over network
- Custom analysis
- etc.
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from frame_classifier import FrameEvent


@runtime_checkable
class FrameHandler(Protocol):
    """Protocol for frame event handlers."""

    def on_frame(self, evt: FrameEvent, pixels: Optional[np.ndarray], info) -> None:
        """Handle a classified frame.

        Args:
            evt: Classification result (category, video_rect, damage_rects, etc.)
            pixels: BGRA pixel data as numpy array, or None for UNCHANGED frames
            info: FrameInfo with tile position and dimensions
        """
        ...

    def on_phase_start(self, name: str) -> None:
        """Called when a new test phase begins.

        Args:
            name: Human-readable phase name
        """
        ...

    def set_drop_count(self, count: int) -> None:
        """Update the current drop count.

        Args:
            count: Cumulative number of dropped frames
        """
        ...

    def close(self) -> None:
        """Clean up resources (flush buffers, close files, etc.)."""
        ...


class BaseFrameHandler(ABC):
    """Abstract base class for frame handlers with default no-op implementations."""

    @abstractmethod
    def on_frame(self, evt: FrameEvent, pixels: Optional[np.ndarray], info) -> None:
        """Handle a classified frame. Must be implemented by subclasses."""
        pass

    def on_phase_start(self, name: str) -> None:
        """Called when a new test phase begins. Override if needed."""
        pass

    def set_drop_count(self, count: int) -> None:
        """Update the current drop count. Override if needed."""
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


class NullHandler(BaseFrameHandler):
    """No-op handler that discards all frames. Useful for benchmarking."""

    def on_frame(self, evt: FrameEvent, pixels: Optional[np.ndarray], info) -> None:
        pass


class PrintHandler(BaseFrameHandler):
    """Simple handler that prints frame info to stdout. Useful for debugging."""

    def __init__(self):
        self._frame_count = 0
        self._phase_name = ""

    def on_phase_start(self, name: str) -> None:
        self._phase_name = name
        print(f"[PrintHandler] Phase: {name}")

    def on_frame(self, evt: FrameEvent, pixels: Optional[np.ndarray], info) -> None:
        self._frame_count += 1
        shape = f"{pixels.shape}" if pixels is not None else "None"

        # Format damage rects
        damage_str = ""
        if evt.damage_rects is not None:
            if len(evt.damage_rects) == 0:
                damage_str = " damage=[]"
            else:
                rects = [f"({r.x},{r.y},{r.width}x{r.height})" for r in evt.damage_rects]
                damage_str = f" damage={rects}"

        # Format video rect for VIDEO_ONLY
        video_str = ""
        if evt.category == "video_only" and evt.video_rect:
            vr = evt.video_rect
            video_str = f" video=({vr[0]},{vr[1]},{vr[2]}x{vr[3]})"

        print(f"  #{self._frame_count} {evt.category:12s} pixels={shape}{damage_str}{video_str}")


class VideoWriterHandler(BaseFrameHandler):
    """Adapter that wraps VerificationVideoWriter as a FrameHandler.

    This is the default handler that produces the color-coded verification
    video with HUD overlay.
    """

    def __init__(self, output_path: str, source_width: int, source_height: int,
                 width: int = 960, height: int = 540, chrome_height: int = 0):
        # Import here to avoid circular dependency
        from video_writer import VerificationVideoWriter

        self._writer = VerificationVideoWriter(
            output_path, source_width, source_height,
            width=width, height=height, chrome_height=chrome_height
        )
        self._writer.open()
        self.output_path = output_path

    def on_phase_start(self, name: str) -> None:
        self._writer.set_phase(name)

    def set_drop_count(self, count: int) -> None:
        self._writer.set_drop_count(count)

    def on_frame(self, evt: FrameEvent, pixels: Optional[np.ndarray], info) -> None:
        self._writer.write_frame(evt, pixels, info)

    def close(self) -> None:
        self._writer.close()
