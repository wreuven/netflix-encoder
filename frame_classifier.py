"""
Frame classifier: combines SHM capture with requestVideoFrameCallback
to track video state and feed it to the Vulkan layer for classification.

The layer reads video state from SHM and classifies each frame:
  UNCHANGED  — no new compositor frame / empty damage rects
  VIDEO_ONLY — only the video element updated (damage within video bounds)
  CHANGED    — DOM/CSS repaint or compositor-level change (scroll, etc.)

Python's role is JS video tracking (requestVideoFrameCallback) and writing
video rect + playing state to SHM. The layer does the actual classification
and encoding atomically.
"""

import json
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pychrome

from shm_reader import (
    DamageRect, FrameCaptureReader, FrameInfo,
    FRAME_CAT_UNCHANGED, FRAME_CAT_VIDEO_ONLY, FRAME_CAT_CHANGED,
)


@dataclass
class FrameEvent:
    """Classification result for a single compositor frame."""
    category: str                      # "unchanged" | "video_only" | "changed"
    video_rect: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h) CSS px
    tile_rect: Optional[Tuple[int, int, int, int]] = None   # (x,y,w,h) px from SHM
    damage_rects: Optional[List[DamageRect]] = None
    timestamp: float = 0.0


_RVFC_JS = r"""
(function() {
    function onVideoFrame(videoId, x, y, w, h, mediaTime) {
        __onVideoFrame(JSON.stringify({videoId: videoId, x: x, y: y, w: w, h: h, mediaTime: mediaTime}));
    }

    function isPopupVideo(v) {
        // Only track videos inside popup containers, not the hero billboard
        // Netflix popup selectors
        var popupSelectors = ['.bob-card', '.mini-modal', '.preview-modal', '.jawBone'];
        for (var i = 0; i < popupSelectors.length; i++) {
            if (v.closest(popupSelectors[i])) return true;
        }
        // Billboard is large (>600px wide) - reject it
        var r = v.getBoundingClientRect();
        if (r.width > 600) return false;
        return false;  // Default: don't track unknown videos
    }

    function trackVideo(v, id) {
        if (v.__rvfcTracked) return;
        v.__rvfcTracked = true;
        function cb(now, metadata) {
            if (!isPopupVideo(v)) {
                v.requestVideoFrameCallback(cb);
                return;
            }
            var r = v.getBoundingClientRect();
            onVideoFrame(id,
                Math.round(r.x), Math.round(r.y),
                Math.round(r.width), Math.round(r.height),
                metadata.mediaTime);
            v.requestVideoFrameCallback(cb);
        }
        v.requestVideoFrameCallback(cb);
    }

    var nextId = 0;
    function scan() {
        document.querySelectorAll('video').forEach(function(v) {
            if (!v.__rvfcId && v.__rvfcId !== 0) v.__rvfcId = nextId++;
            trackVideo(v, v.__rvfcId);
        });
    }
    scan();
    new MutationObserver(scan).observe(document, {childList: true, subtree: true});
})()
"""


@dataclass
class _VideoFrameEvent:
    """Internal: a video frame callback from JS."""
    video_id: int
    x: int
    y: int
    w: int
    h: int
    media_time: float
    timestamp: float  # Python monotonic time when received


class FrameClassifier:
    """
    Track video state via requestVideoFrameCallback and write it to SHM.
    The Vulkan layer reads this state and classifies frames.
    """

    def __init__(self, tab: pychrome.Tab, reader: FrameCaptureReader):
        self.tab = tab
        self.reader = reader

        self._lock = threading.Lock()
        self._running = False

        # Viewport size (device pixels) — set from SHM reader
        vp = reader.get_frame_dimensions()
        self._viewport_w = vp[0]
        self._viewport_h = vp[1]

        self._dpr = 1.0
        self._chrome_height = 0  # browser chrome offset (CSS px)

        # Video frame events from JS binding
        self._video_events: List[_VideoFrameEvent] = []

        # Recent video state for vsync-repeat classification
        self._last_video_rect: Optional[Tuple[int, int, int, int]] = None
        self._last_video_time: float = 0.0

    def start(self):
        """Enable tracking: JS video callback binding, SHM."""
        self._running = True

        # Query device pixel ratio and browser chrome height
        try:
            r = self.tab.Runtime.evaluate(
                expression="window.devicePixelRatio", returnByValue=True)
            self._dpr = float(r.get("result", {}).get("value", 1.0))
        except Exception:
            pass
        try:
            r = self.tab.Runtime.evaluate(
                expression="window.outerHeight - window.innerHeight",
                returnByValue=True)
            self._chrome_height = int(r.get("result", {}).get("value", 0))
        except Exception:
            pass

        # Set up Runtime.bindingCalled handler and add binding
        self.tab.Runtime.bindingCalled = self._on_binding_called
        try:
            self.tab.Runtime.addBinding(name="__onVideoFrame")
        except Exception:
            pass

        # Inject the requestVideoFrameCallback tracker JS
        try:
            self.tab.Runtime.evaluate(expression=_RVFC_JS, returnByValue=True)
        except Exception as e:
            print(f"  [FrameClassifier] Warning: JS injection failed: {e}")

        # Start periodic any_video_playing check (every ~500ms)
        self._last_video_playing_check = 0.0
        self._video_playing_interval = 0.5

    def _on_binding_called(self, **kwargs):
        """CDP Runtime.bindingCalled event handler for __onVideoFrame."""
        if not self._running:
            return
        name = kwargs.get("name", "")
        if name != "__onVideoFrame":
            return
        payload = kwargs.get("payload", "")
        try:
            data = json.loads(payload)
            evt = _VideoFrameEvent(
                video_id=int(data.get("videoId", 0)),
                x=int(data.get("x", 0)),
                y=int(data.get("y", 0)),
                w=int(data.get("w", 0)),
                h=int(data.get("h", 0)),
                media_time=float(data.get("mediaTime", 0)),
                timestamp=time.monotonic(),
            )
            with self._lock:
                self._video_events.append(evt)

            # Write video state to SHM immediately
            dx = int(evt.x * self._dpr)
            dy = int((evt.y + self._chrome_height) * self._dpr)
            dw = int(evt.w * self._dpr)
            dh = int(evt.h * self._dpr)
            self.reader.set_video_rect(dx, dy, dw, dh)
            self.reader.set_video_last_callback_ns(time.monotonic_ns())
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    def _drain_video_events(self) -> List[_VideoFrameEvent]:
        """Return and clear pending video frame events."""
        with self._lock:
            events = self._video_events
            self._video_events = []
            return events

    def _any_video_playing(self) -> bool:
        """Check if any video element is currently playing on the page.
        Also writes the result to SHM for the layer's classification logic."""
        try:
            result = self.tab.Runtime.evaluate(
                expression="Array.from(document.querySelectorAll('video')).some(v => !v.paused && !v.ended)",
                returnByValue=True,
            )
            playing = bool(result.get("result", {}).get("value", False))
            self.reader.set_any_video_playing(playing)
            return playing
        except Exception:
            return False

    def _maybe_check_video_playing(self):
        """Periodically check if any video is playing and write to SHM."""
        now = time.monotonic()
        if now - self._last_video_playing_check >= self._video_playing_interval:
            self._last_video_playing_check = now
            self._any_video_playing()

    def next_frame_v4(self, timeout: float = 0.1):
        """
        Read frame from SHM (no pixels). Classification done by layer.

        Drains video events (updates SHM video state), reads frame header +
        category + bitstreams from SHM.

        Returns (FrameEvent, FrameInfo) or (FrameEvent(unchanged), None) if no frame.
        """
        # Drain video events to keep SHM video state updated
        video_events = self._drain_video_events()
        if video_events:
            latest = video_events[-1]
            self._last_video_rect = (latest.x, latest.y, latest.w, latest.h)
            self._last_video_time = time.monotonic()

        # Periodic video playing check
        self._maybe_check_video_playing()

        result = self.reader.read_frame_v4(timeout=timeout)

        if result is None:
            return (
                FrameEvent(category="unchanged", timestamp=time.monotonic()),
                None,
            )

        info, category, enc1_bs, enc2_bs, enc2_region = result

        # Map category int to string
        cat_map = {
            FRAME_CAT_UNCHANGED: "unchanged",
            FRAME_CAT_VIDEO_ONLY: "video_only",
            FRAME_CAT_CHANGED: "changed",
        }
        cat_str = cat_map.get(category, "changed")

        evt = FrameEvent(
            category=cat_str,
            video_rect=self._last_video_rect,
            tile_rect=(info.x, info.y, info.width, info.height),
            damage_rects=info.damage_rects,
            timestamp=time.monotonic(),
        )

        # Attach v4 bitstream data to info for the handler
        info.enc1_bitstream = enc1_bs
        info.enc2_bitstream = enc2_bs
        info.enc2_region = enc2_region

        return (evt, info)

    def stop(self):
        """Stop classification."""
        self._running = False
