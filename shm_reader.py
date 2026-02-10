"""
Shared memory reader for VK_LAYER_CHROME_frame_capture.

Reads pixel data from the POSIX shared memory region populated by the
Vulkan capture layer.  Uses struct to mirror the C shm_header_t layout.
"""

import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Constants matching shm_protocol.h
SHM_MAGIC = 0x56464341
SHM_VERSION = 4
SHM_NAME = "/chrome_frame_capture"
SHM_PATH = "/dev/shm" + SHM_NAME
SHM_HEADER_SIZE = 4096
MAX_FRAME_WIDTH = 3840
MAX_FRAME_HEIGHT = 2160
BYTES_PER_PIXEL = 4
SHM_MAX_DATA_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * BYTES_PER_PIXEL
SHM_BITSTREAM_OFFSET = SHM_HEADER_SIZE + SHM_MAX_DATA_SIZE
SHM_ENC1_BITSTREAM_OFFSET = SHM_BITSTREAM_OFFSET
SHM_ENC1_BITSTREAM_MAX = 1024 * 1024  # 1MB
SHM_ENC2_BITSTREAM_OFFSET = SHM_BITSTREAM_OFFSET + SHM_ENC1_BITSTREAM_MAX
SHM_ENC2_BITSTREAM_MAX = 1024 * 1024  # 1MB
SHM_BITSTREAM_MAX = SHM_ENC1_BITSTREAM_MAX + SHM_ENC2_BITSTREAM_MAX
SHM_TOTAL_SIZE = SHM_BITSTREAM_OFFSET + SHM_BITSTREAM_MAX

# Frame category constants (match shm_protocol.h)
FRAME_CAT_UNCHANGED = 0
FRAME_CAT_VIDEO_ONLY = 1
FRAME_CAT_CHANGED = 2

# Damage rects
DAMAGE_RECT_NOT_PRESENT = 0xFFFFFFFF
MAX_DAMAGE_RECTS = 32

# Flags
FLAG_LAYER_READY = 1 << 0
FLAG_FRAME_AVAILABLE = 1 << 1
FLAG_CAPTURE_SUBRECT = 1 << 2
FLAG_READER_BUSY = 1 << 3
FLAG_SHUTDOWN = 1 << 4
FLAG_COPY_PIXELS = 1 << 5

# Encoder control field offsets (v3+)
# encode_request at offset 624 (after damage_rects[32] @ 112..624)
_OFF_ENCODE_REQUEST = 624
_OFF_ENCODER_CONFIG = 628
_OFF_BITSTREAM_SIZE = 632

# v4 field offsets
_OFF_VIDEO_RECT_X = 640
_OFF_VIDEO_RECT_Y = 644
_OFF_VIDEO_RECT_W = 648
_OFF_VIDEO_RECT_H = 652
_OFF_VIDEO_LAST_CALLBACK_NS = 656
_OFF_ANY_VIDEO_PLAYING = 664

_OFF_FRAME_CATEGORY = 672
_OFF_ENC2_BITSTREAM_SIZE = 676
_OFF_ENC2_REGION_MB_X = 680
_OFF_ENC2_REGION_MB_Y = 684
_OFF_ENC2_REGION_MB_W = 688
_OFF_ENC2_REGION_MB_H = 692

# struct layout of shm_header_t fields we access (excluding reserved[])
# See shm_protocol.h for definitive layout.
#
# Offsets (bytes):
#   0   magic            u32
#   4   version          u32
#   8   flags            u32
#  12   _pad0            u32
#  16   frame_width      u32
#  20   frame_height     u32
#  24   frame_format     u32
#  28   frame_stride     u32
#  32   sub_x            i32
#  36   sub_y            i32
#  40   sub_width        u32
#  44   sub_height       u32
#  48   captured_x       i32
#  52   captured_y       i32
#  56   captured_width   u32
#  60   captured_height  u32
#  64   captured_stride  u32
#  68   _pad1            u32
#  72   write_seq        u64
#  80   read_seq         u64
#  88   frame_timestamp  u64
#  96   present_count    u32
# 100   skip_count       u32

_HEADER_FMT = "<IIIIIIIIiiIIiiIIIIQQQII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # should be 104

# Named field indices
_F_MAGIC = 0
_F_VERSION = 1
_F_FLAGS = 2
_F_FRAME_W = 4
_F_FRAME_H = 5
_F_FRAME_FMT = 6
_F_FRAME_STRIDE = 7
_F_SUB_X = 8
_F_SUB_Y = 9
_F_SUB_W = 10
_F_SUB_H = 11
_F_CAP_X = 12
_F_CAP_Y = 13
_F_CAP_W = 14
_F_CAP_H = 15
_F_CAP_STRIDE = 16
_F_WRITE_SEQ = 18
_F_READ_SEQ = 19
_F_TIMESTAMP = 20
_F_PRESENT = 21
_F_SKIP_COUNT = 22


@dataclass
class DamageRect:
    """A damage rectangle from VK_KHR_incremental_present."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class FrameInfo:
    """Metadata for a captured frame."""
    width: int
    height: int
    stride: int
    x: int
    y: int
    format: int
    timestamp_ns: int
    present_count: int
    write_seq: int
    damage_rects: Optional[List[DamageRect]] = None
    # v4 fields (set by frame_classifier.next_frame_v4)
    enc1_bitstream: Optional[bytes] = None
    enc2_bitstream: Optional[bytes] = None
    enc2_region: Optional[tuple] = None  # (mb_x, mb_y, mb_w, mb_h)


class FrameCaptureReader:
    """Reads frames from the Vulkan layer's shared memory region."""

    def __init__(self):
        self._fd: int = -1
        self._mm: Optional[mmap.mmap] = None
        self._last_seq: int = 0
        self._connected = False
        self._has_damage_rects = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, timeout: float = 10.0) -> bool:
        """Open shared memory and wait for the layer to be ready."""
        start = time.monotonic()

        # Wait for shm file to appear
        while time.monotonic() - start < timeout:
            if os.path.exists(SHM_PATH):
                break
            time.sleep(0.1)
        else:
            return False

        self._fd = os.open(SHM_PATH, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, SHM_TOTAL_SIZE,
                             mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE)

        # Wait for LAYER_READY
        while time.monotonic() - start < timeout:
            hdr = self._read_header()
            if hdr[_F_MAGIC] == SHM_MAGIC and (hdr[_F_FLAGS] & FLAG_LAYER_READY):
                self._connected = True
                self._has_damage_rects = hdr[_F_VERSION] >= 2
                # Clear SHUTDOWN flag in case a previous reader set it
                self._clear_flags(FLAG_SHUTDOWN)
                return True
            time.sleep(0.1)

        return False

    def get_frame_dimensions(self) -> tuple:
        """Return (width, height) of the full swapchain frame."""
        hdr = self._read_header()
        return (hdr[_F_FRAME_W], hdr[_F_FRAME_H])

    def get_skip_count(self) -> int:
        """Return the number of presents the layer skipped (reader too slow)."""
        hdr = self._read_header()
        return hdr[_F_SKIP_COUNT]

    def set_capture_rect(self, x: int, y: int, w: int, h: int):
        """Tell the layer to capture a sub-rectangle next frame."""
        if not self._connected:
            return
        struct.pack_into("<iiII", self._mm, 32, x, y, w, h)
        self._or_flags(FLAG_CAPTURE_SUBRECT)

    def set_capture_full(self):
        """Tell the layer to capture the full frame."""
        if not self._connected:
            return
        self._clear_flags(FLAG_CAPTURE_SUBRECT)

    def read_frame(self, timeout: float = 0.1) -> Optional[tuple]:
        """
        Read the next available frame.

        Returns:
            (numpy_array, FrameInfo) or None if no new frame within timeout.
            numpy_array is shape (H, W, 4) dtype uint8 in BGRA order.
        """
        if not self._connected:
            return None

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            hdr = self._read_header()
            if hdr[_F_WRITE_SEQ] > self._last_seq:
                break
            time.sleep(0.0001)  # 100us poll
        else:
            return None

        # Signal that we are reading
        self._or_flags(FLAG_READER_BUSY)
        self._clear_flags(FLAG_FRAME_AVAILABLE)

        hdr = self._read_header()
        w = hdr[_F_CAP_W]
        h = hdr[_F_CAP_H]
        stride = hdr[_F_CAP_STRIDE]

        if w == 0 or h == 0:
            self._clear_flags(FLAG_READER_BUSY)
            return None

        row_bytes = w * BYTES_PER_PIXEL

        # Read pixel data — create a numpy view directly on the mmap
        # and .copy() once, avoiding the intermediate bytes/bytearray.
        if stride == row_bytes or stride == 0:
            arr = np.ndarray(
                (h, w, 4), dtype=np.uint8,
                buffer=self._mm, offset=SHM_HEADER_SIZE,
            ).copy()
        else:
            # Strided layout: use numpy strides to skip padding bytes
            arr = np.ndarray(
                (h, w, 4), dtype=np.uint8,
                buffer=self._mm, offset=SHM_HEADER_SIZE,
                strides=(stride, BYTES_PER_PIXEL, 1),
            ).copy()

        damage = self._read_damage_rects()

        info = FrameInfo(
            width=w, height=h, stride=stride,
            x=hdr[_F_CAP_X], y=hdr[_F_CAP_Y],
            format=hdr[_F_FRAME_FMT],
            timestamp_ns=hdr[_F_TIMESTAMP],
            present_count=hdr[_F_PRESENT],
            write_seq=hdr[_F_WRITE_SEQ],
            damage_rects=damage,
        )

        self._last_seq = hdr[_F_WRITE_SEQ]

        # Update read_seq
        struct.pack_into("<Q", self._mm, 80, self._last_seq)

        self._clear_flags(FLAG_READER_BUSY)

        return arr, info

    def _read_damage_rects(self) -> Optional[list]:
        """Read damage rects from the SHM header (v2+)."""
        if not self._has_damage_rects:
            return None
        count = struct.unpack_from("<I", self._mm, 104)[0]
        if count == DAMAGE_RECT_NOT_PRESENT:
            return None
        if count == 0:
            return []
        n = min(count, MAX_DAMAGE_RECTS)
        rects = []
        for i in range(n):
            offset = 112 + i * 16
            x, y, w, h = struct.unpack_from("<iiII", self._mm, offset)
            rects.append(DamageRect(x=x, y=y, width=w, height=h))
        return rects

    # ---- Encoder control (v3+) ----

    def set_encode_request(self, request: int):
        """Set encode_request for the next frame (0=none, 1=encode, 2=force_idr)."""
        if self._mm:
            struct.pack_into("<I", self._mm, _OFF_ENCODE_REQUEST, request)

    def set_encoder_config(self, qp: int):
        """Set encoder QP config (low 8 bits)."""
        if self._mm:
            struct.pack_into("<I", self._mm, _OFF_ENCODER_CONFIG, qp & 0xFF)

    def get_bitstream_size(self) -> int:
        """Read bitstream_size from SHM header (0 if no bitstream)."""
        if not self._mm:
            return 0
        return struct.unpack_from("<I", self._mm, _OFF_BITSTREAM_SIZE)[0]

    def read_bitstream(self) -> Optional[bytes]:
        """Read encoded bitstream from SHM bitstream region, or None."""
        if not self._mm:
            return None
        bs_size = struct.unpack_from("<I", self._mm, _OFF_BITSTREAM_SIZE)[0]
        if bs_size == 0:
            return None
        return bytes(self._mm[SHM_BITSTREAM_OFFSET:SHM_BITSTREAM_OFFSET + bs_size])

    # ---- v4: Video state writers (Python → layer) ----

    def set_video_rect(self, x: int, y: int, w: int, h: int):
        """Write video rect in device pixels to SHM."""
        if self._mm:
            struct.pack_into("<iiII", self._mm, _OFF_VIDEO_RECT_X, x, y, w, h)

    def set_video_last_callback_ns(self, ns: int):
        """Write video callback timestamp (CLOCK_MONOTONIC ns) to SHM."""
        if self._mm:
            struct.pack_into("<Q", self._mm, _OFF_VIDEO_LAST_CALLBACK_NS, ns)

    def set_any_video_playing(self, playing: bool):
        """Write any_video_playing flag to SHM."""
        if self._mm:
            struct.pack_into("<I", self._mm, _OFF_ANY_VIDEO_PLAYING, 1 if playing else 0)

    # ---- v4: Classification + encoder2 readers (layer → Python) ----

    def get_frame_category(self) -> int:
        """Read frame_category from SHM header."""
        if not self._mm:
            return FRAME_CAT_UNCHANGED
        return struct.unpack_from("<I", self._mm, _OFF_FRAME_CATEGORY)[0]

    def read_encoder1_bitstream(self) -> Optional[bytes]:
        """Read encoder1 bitstream from SHM, or None."""
        if not self._mm:
            return None
        bs_size = struct.unpack_from("<I", self._mm, _OFF_BITSTREAM_SIZE)[0]
        if bs_size == 0:
            return None
        return bytes(self._mm[SHM_ENC1_BITSTREAM_OFFSET:SHM_ENC1_BITSTREAM_OFFSET + bs_size])

    def read_encoder2_bitstream(self) -> Optional[bytes]:
        """Read encoder2 bitstream from SHM, or None."""
        if not self._mm:
            return None
        bs_size = struct.unpack_from("<I", self._mm, _OFF_ENC2_BITSTREAM_SIZE)[0]
        if bs_size == 0:
            return None
        return bytes(self._mm[SHM_ENC2_BITSTREAM_OFFSET:SHM_ENC2_BITSTREAM_OFFSET + bs_size])

    def get_encoder2_region(self) -> tuple:
        """Read encoder2 region MB coords: (mb_x, mb_y, mb_w, mb_h)."""
        if not self._mm:
            return (0, 0, 0, 0)
        return struct.unpack_from("<iiII", self._mm, _OFF_ENC2_REGION_MB_X)

    def read_frame_v4(self, timeout: float = 0.1):
        """
        Read next frame in v4 mode: header + damage_rects + category + bitstreams.
        NO pixel copy.

        Returns:
            (FrameInfo, category, enc1_bs, enc2_bs, enc2_region) or None if no new frame.
            enc2_region is (mb_x, mb_y, mb_w, mb_h).
        """
        if not self._connected:
            return None

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            hdr = self._read_header()
            if hdr[_F_WRITE_SEQ] > self._last_seq:
                break
            time.sleep(0.0001)
        else:
            return None

        # Signal that we are reading
        self._or_flags(FLAG_READER_BUSY)
        self._clear_flags(FLAG_FRAME_AVAILABLE)

        hdr = self._read_header()
        w = hdr[_F_CAP_W]
        h = hdr[_F_CAP_H]
        stride = hdr[_F_CAP_STRIDE]

        if w == 0 or h == 0:
            self._clear_flags(FLAG_READER_BUSY)
            return None

        damage = self._read_damage_rects()
        category = self.get_frame_category()
        enc1_bs = self.read_encoder1_bitstream()
        enc2_bs = self.read_encoder2_bitstream()
        enc2_region = self.get_encoder2_region()

        info = FrameInfo(
            width=w, height=h, stride=stride,
            x=hdr[_F_CAP_X], y=hdr[_F_CAP_Y],
            format=hdr[_F_FRAME_FMT],
            timestamp_ns=hdr[_F_TIMESTAMP],
            present_count=hdr[_F_PRESENT],
            write_seq=hdr[_F_WRITE_SEQ],
            damage_rects=damage,
        )

        self._last_seq = hdr[_F_WRITE_SEQ]
        struct.pack_into("<Q", self._mm, 80, self._last_seq)

        self._clear_flags(FLAG_READER_BUSY)

        return (info, category, enc1_bs, enc2_bs, enc2_region)

    def disconnect(self):
        """Signal shutdown and release resources."""
        if self._connected:
            self._or_flags(FLAG_SHUTDOWN)
            self._connected = False
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    # ---- internal ----

    def _read_header(self) -> tuple:
        raw = self._mm[0:_HEADER_SIZE]
        return struct.unpack(_HEADER_FMT, raw)

    def _or_flags(self, bits: int):
        """Atomically OR bits into the flags field (offset 8)."""
        current = struct.unpack_from("<I", self._mm, 8)[0]
        struct.pack_into("<I", self._mm, 8, current | bits)

    def _clear_flags(self, bits: int):
        """Atomically clear bits from the flags field."""
        current = struct.unpack_from("<I", self._mm, 8)[0]
        struct.pack_into("<I", self._mm, 8, current & ~bits)
