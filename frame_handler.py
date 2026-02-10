"""
Frame handler base class for pluggable frame processing.

Handlers receive classified frame events and bitstreams from the Vulkan
layer's dual NVENC encoders. In SHM v4, pixels are never copied — handlers
work with pre-encoded H.264 bitstreams only.
"""

from abc import ABC, abstractmethod
from typing import Optional

from frame_classifier import FrameEvent


class BaseFrameHandler(ABC):
    """Abstract base class for frame handlers."""

    @abstractmethod
    def on_frame(self, evt: FrameEvent, pixels, info) -> None:
        """Handle a classified frame.

        Args:
            evt: Classification result (category, video_rect, damage_rects, etc.)
            pixels: Always None in v4 (no pixel copy)
            info: FrameInfo with bitstreams and region metadata
        """
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
