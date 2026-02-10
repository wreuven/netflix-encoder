"""
Frame classifier: combines SHM capture with Vulkan damage rects from
VK_KHR_incremental_present and requestVideoFrameCallback to classify
each compositor frame.

Categories:
  UNCHANGED  — no new compositor frame from SHM / empty damage rects
  VIDEO_ONLY — only the video element updated (damage within video bounds)
  CHANGED    — DOM/CSS repaint or compositor-level change (scroll, etc.)

The Vulkan layer extracts damage rects from VkPresentRegionsKHR and
writes them into the SHM header.  These are in swapchain (device pixel)
coordinates which include browser chrome, so a chrome_height offset is
applied when comparing against CSS video rects from
getBoundingClientRect().
"""

import json
import sys
import threading
import time
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
    Classify compositor frames using two signals:

    1. Vulkan damage rects from VK_KHR_incremental_present (via SHM)
    2. requestVideoFrameCallback — a video frame was composited
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

            # v4: Write video state to SHM immediately
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

    def next_frame(self, timeout: float = 0.1) -> Optional[FrameEvent]:
        """
        Poll SHM for the next frame and classify it.

        Returns FrameEvent or None if no new frame within timeout.
        """
        result = self.reader.read_frame(timeout=timeout)

        video_events = self._drain_video_events()

        if result is None:
            return FrameEvent(
                category="unchanged",
                timestamp=time.monotonic(),
            )

        arr, info = result
        tile_rect = (info.x, info.y, info.width, info.height)

        return self._classify(info.damage_rects, video_events, tile_rect)

    def next_frame_with_pixels(self, timeout: float = 0.1):
        """
        Like next_frame() but also returns raw pixel data.

        Returns (FrameEvent, numpy_array | None, FrameInfo | None).
        """
        result = self.reader.read_frame(timeout=timeout)

        video_events = self._drain_video_events()

        if result is None:
            return (
                FrameEvent(category="unchanged", timestamp=time.monotonic()),
                None,
                None,
            )

        arr, info = result
        tile_rect = (info.x, info.y, info.width, info.height)
        evt = self._classify(info.damage_rects, video_events, tile_rect)
        return evt, arr, info

    @staticmethod
    def _damage_within_video_rect(damage_rects: List[DamageRect],
                                  video_rect: Tuple[int, int, int, int],
                                  dpr: float,
                                  chrome_height: int = 0,
                                  tolerance: int = 4) -> bool:
        """Check if all damage rects fall within the video rect.

        Converts CSS video_rect to device (swapchain) pixels via DPR and
        chrome_height offset.  CSS getBoundingClientRect() is relative to
        the viewport (below browser chrome), while Vulkan damage rects are
        in swapchain coordinates (includes browser chrome).
        """
        vx = int(video_rect[0] * dpr) - tolerance
        vy = int((video_rect[1] + chrome_height) * dpr) - tolerance
        vr = int((video_rect[0] + video_rect[2]) * dpr) + tolerance
        vb = int((video_rect[1] + chrome_height + video_rect[3]) * dpr) + tolerance

        for dr in damage_rects:
            if dr.x < vx or dr.y < vy:
                return False
            if dr.x + int(dr.width) > vr or dr.y + int(dr.height) > vb:
                return False
        return True

    def _classify(self, damage_rects, video_events, tile_rect):
        """Classify a frame based on damage rects and video events.

        Chrome's display compositor (viz) sends exactly one VkRectLayerKHR
        per present — the bounding box of all dirty regions.  When only
        the video texture updates, this bbox equals the video element rect
        exactly.  When anything else also changes (DOM repaint, hover
        state, row animation), the bbox grows to encompass both, so the
        containment check correctly detects the non-video change.

        With damage rects (Vulkan VK_KHR_incremental_present):
        | damage_rects | video callback            | Result     |
        |--------------|---------------------------|------------|
        | empty        | no                        | UNCHANGED  |
        | empty        | yes or recent (<200ms)    | VIDEO_ONLY |
        | non-empty    | all within video_rect     | VIDEO_ONLY |
        | non-empty    | any outside video_rect    | CHANGED    |

        Without damage rects (X11 / extension absent):
        | video callback            | Result     |
        |---------------------------|------------|
        | yes or recent (<200ms)    | VIDEO_ONLY |
        | no                        | CHANGED    |
        """
        now = time.monotonic()

        if video_events:
            latest = video_events[-1]
            self._last_video_rect = (latest.x, latest.y, latest.w, latest.h)
            self._last_video_time = now

        # damage_rects is None means the Vulkan layer is not working
        if damage_rects is None:
            raise RuntimeError(
                "Damage rects not available — Vulkan layer not loaded. "
                "Ensure Chrome is launched with VK_LAYER_PATH and CHROME_FRAME_CAPTURE=1"
            )

        # Empty damage rects: nothing changed on the Vulkan side
        if len(damage_rects) == 0:
            recent_video = (self._last_video_rect is not None
                            and now - self._last_video_time < 0.2)
            if video_events or recent_video:
                return FrameEvent(
                    category="video_only",
                    video_rect=self._last_video_rect,
                    tile_rect=tile_rect,
                    damage_rects=damage_rects,
                    timestamp=now,
                )
            # Check if any video is playing on the page (including hero)
            # If so, treat as CHANGED to ensure we capture those frames
            if self._any_video_playing():
                return FrameEvent(
                    category="changed",
                    tile_rect=tile_rect,
                    damage_rects=damage_rects,
                    timestamp=now,
                )
            return FrameEvent(
                category="unchanged",
                tile_rect=tile_rect,
                damage_rects=damage_rects,
                timestamp=now,
            )

        # Non-empty damage rects: check if all within video rect
        video_rect = self._last_video_rect
        if video_rect and (video_events or
                           now - self._last_video_time < 0.2):
            if self._damage_within_video_rect(damage_rects, video_rect,
                                              self._dpr,
                                              self._chrome_height):
                return FrameEvent(
                    category="video_only",
                    video_rect=video_rect,
                    tile_rect=tile_rect,
                    damage_rects=damage_rects,
                    timestamp=now,
                )

        # Damage rects exist outside video area → CHANGED
        return FrameEvent(
            category="changed",
            tile_rect=tile_rect,
            damage_rects=damage_rects,
            timestamp=now,
        )

    def _maybe_check_video_playing(self):
        """Periodically check if any video is playing and write to SHM."""
        now = time.monotonic()
        if now - self._last_video_playing_check >= self._video_playing_interval:
            self._last_video_playing_check = now
            self._any_video_playing()

    def next_frame_v4(self, timeout: float = 0.1):
        """
        v4: Read frame from SHM (no pixels). Classification done by layer.

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
