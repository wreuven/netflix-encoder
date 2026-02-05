#!/usr/bin/env python3
"""
POC: VFR via PyAV demux/remux with timestamp modification.

Instead of creating packets from NAL bytes, we:
1. Write raw H.264 to file
2. Open it with PyAV for reading
3. Remux to MKV with modified timestamps
"""

import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, "/home/wachtfogel/h264-newpoc/splicer")
from nvenc_encoder import NVENCEncoder

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_test_frame(width, height, frame_num):
    """Create a simple test frame."""
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    color = (frame_num * 10) % 256
    frame[:, :, 0] = color
    frame[:, :, 1] = (255 - color)
    frame[:, :, 2] = 128
    frame[:, :, 3] = 255
    box_x = (frame_num * 20) % (width - 100)
    frame[100:200, box_x:box_x+100, :3] = 255
    return frame


def bgra_to_nv12(bgra):
    """Convert BGRA to NV12."""
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

    h264_path = OUTPUT_DIR / "vfr_test2.h264"
    mkv_path = OUTPUT_DIR / "vfr_test2.mkv"

    # Frames to encode (skipping some to test VFR)
    frames_to_encode = [0, 1, 2, 5, 6, 8, 9, 10, 11, 12]

    print(f"Step 1: Encode {len(frames_to_encode)} frames to raw H.264...")

    with open(h264_path, "wb") as f:
        for i, input_frame_num in enumerate(frames_to_encode):
            pixels = create_test_frame(width, height, input_frame_num)
            nv12 = bgra_to_nv12(pixels)
            force_idr = (i == 0)
            bitstream = encoder.encode(nv12, force_idr=force_idr)
            f.write(bitstream)
            print(f"  Frame {i}: input={input_frame_num}")

    encoder.close()
    print(f"  Written: {h264_path}")

    # Verify H.264 is valid
    print("\nStep 2: Verify H.264...")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1", str(h264_path)
    ], capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")

    # Remux to MKV with VFR timestamps using PyAV
    print("\nStep 3: Remux to MKV with VFR timestamps via PyAV...")

    input_container = av.open(str(h264_path))
    input_stream = input_container.streams.video[0]

    output_container = av.open(str(mkv_path), 'w')
    output_stream = output_container.add_stream('h264')
    output_stream.width = input_stream.width
    output_stream.height = input_stream.height
    output_stream.time_base = input_stream.time_base
    if input_stream.codec_context.extradata:
        output_stream.codec_context.extradata = input_stream.codec_context.extradata

    print(f"  Input time_base: {input_stream.time_base}")
    print(f"  Output time_base: {output_stream.time_base}")

    frame_idx = 0
    for packet in input_container.demux(input_stream):
        # Raw H.264 has no timestamps - we need to assign them
        if frame_idx < len(frames_to_encode):
            input_frame_num = frames_to_encode[frame_idx]
            # Convert frame number to timebase units
            # If time_base is 1/1200000 and we want 25fps, each frame is 48000 units
            frame_duration = int(input_stream.time_base.denominator / 25)
            pts = input_frame_num * frame_duration
            packet.pts = pts
            packet.dts = pts
            print(f"  Packet {frame_idx}: size={packet.size}, pts={pts}, input_frame={input_frame_num}")
            frame_idx += 1
        else:
            continue

        packet.stream = output_stream
        try:
            output_container.mux(packet)
        except Exception as e:
            print(f"  ERROR muxing packet: {e}")

    print(f"  Total packets muxed: {frame_idx}")

    output_container.close()
    input_container.close()

    # Check file was created
    import os
    if os.path.exists(mkv_path):
        print(f"  File size: {os.path.getsize(mkv_path)} bytes")
    else:
        print(f"  ERROR: File not created!")

    print(f"  Written: {mkv_path}")

    # Verify MKV
    print("\nStep 4: Verify MKV...")
    result = subprocess.run([
        "ffprobe", "-v", "warning", str(mkv_path)
    ], capture_output=True, text=True)
    if result.stderr:
        print(f"  Warnings: {result.stderr[:500]}")
    else:
        print("  No errors!")

    print(f"\nTest with: ffplay {mkv_path}")


if __name__ == "__main__":
    main()
