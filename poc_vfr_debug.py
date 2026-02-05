#!/usr/bin/env python3
"""
Debug: Compare packet data between demux and direct approaches.
"""

import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, "/home/wachtfogel/h264-newpoc/splicer")
from nvenc_encoder import NVENCEncoder
from h264_splicer import parse_annexb

OUTPUT_DIR = Path(__file__).parent / "output"


def create_test_frame(width, height, frame_num):
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    color = (frame_num * 10) % 256
    frame[:, :, 0] = color
    frame[:, :, 1] = (255 - color)
    frame[:, :, 2] = 128
    frame[:, :, 3] = 255
    return frame


def bgra_to_nv12(bgra):
    b = bgra[:, :, 0].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    r = bgra[:, :, 2].astype(np.float32)
    y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
    u = ((-0.169 * r - 0.331 * g + 0.500 * b) + 128).clip(0, 255).astype(np.uint8)
    v = ((0.500 * r - 0.419 * g - 0.081 * b) + 128).clip(0, 255).astype(np.uint8)
    height, width = y.shape
    u_sub = u[::2, ::2]
    v_sub = v[::2, ::2]
    uv = np.zeros((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u_sub
    uv[:, 1::2] = v_sub
    return np.vstack([y, uv])


def main():
    width, height = 640, 480
    encoder = NVENCEncoder(width, height, qp=23)

    h264_path = OUTPUT_DIR / "vfr_debug.h264"

    # Encode 3 frames
    print("Encoding 3 frames...")
    with open(h264_path, "wb") as f:
        for i in range(3):
            pixels = create_test_frame(width, height, i)
            nv12 = bgra_to_nv12(pixels)
            bitstream = encoder.encode(nv12, force_idr=(i == 0))
            f.write(bitstream)
            print(f"  Frame {i}: {len(bitstream)} bytes")

    encoder.close()

    # Now compare: demux packets vs our manual NAL parsing
    print("\n--- Demuxing with PyAV ---")
    container = av.open(str(h264_path))
    stream = container.streams.video[0]
    print(f"Stream time_base: {stream.time_base}")
    print(f"Extradata: {len(stream.codec_context.extradata) if stream.codec_context.extradata else 0} bytes")
    if stream.codec_context.extradata:
        print(f"Extradata hex: {stream.codec_context.extradata[:40].hex()}...")

    for i, pkt in enumerate(container.demux(stream)):
        if i >= 3:
            break
        print(f"\nPacket {i}:")
        print(f"  size: {pkt.size}")
        print(f"  pts: {pkt.pts}, dts: {pkt.dts}")
        print(f"  is_keyframe: {pkt.is_keyframe}")
        data = bytes(pkt)
        print(f"  first 20 bytes: {data[:20].hex()}")
        # Check if it's Annex B (starts with 00 00 00 01) or AVCC (length prefix)
        if data[:4] == b'\x00\x00\x00\x01':
            print(f"  format: Annex B (start code)")
        else:
            length = int.from_bytes(data[:4], 'big')
            print(f"  format: AVCC (length={length})")

    container.close()

    # Now our manual parsing
    print("\n--- Manual NAL parsing ---")
    with open(h264_path, "rb") as f:
        bitstream = f.read()

    nals = parse_annexb(bitstream)
    print(f"Total NALs: {len(nals)}")

    # Group into AUs
    aus = []
    current_au = []
    for nal in nals:
        if nal.nal_unit_type in (7, 8):
            current_au.append(nal)
        elif nal.nal_unit_type in (1, 5):
            from h264_splicer import BitReader
            reader = BitReader(nal.rbsp)
            first_mb = reader.read_ue()
            if first_mb == 0 and current_au and any(n.nal_unit_type in (1, 5) for n in current_au):
                aus.append(current_au)
                current_au = []
            current_au.append(nal)
    if current_au:
        aus.append(current_au)

    print(f"Access Units: {len(aus)}")

    for i, au in enumerate(aus[:3]):
        print(f"\nAU {i}:")
        for nal in au:
            nal_bytes = nal.to_bytes()
            print(f"  NAL type={nal.nal_unit_type}, size={len(nal_bytes)}, first bytes: {nal_bytes[:10].hex()}")

        # Build Annex B frame data (slices only)
        frame_data = b''
        for nal in au:
            if nal.nal_unit_type in (1, 5):
                frame_data += b'\x00\x00\x00\x01' + nal.to_bytes()
        print(f"  Frame data (slices): {len(frame_data)} bytes, first 20: {frame_data[:20].hex()}")


if __name__ == "__main__":
    main()
