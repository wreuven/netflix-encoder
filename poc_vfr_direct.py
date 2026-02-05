#!/usr/bin/env python3
"""
POC: Direct VFR MKV output - write frames one at a time with custom PTS.
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

    mkv_path = OUTPUT_DIR / "vfr_direct.mkv"

    # Frames to encode (skipping some to test VFR)
    frames_to_encode = [0, 1, 2, 5, 6, 8, 9, 10, 11, 12]

    print(f"Direct VFR output: {len(frames_to_encode)} frames to MKV...")

    # Open output MKV
    out = av.open(str(mkv_path), mode="w")
    st = out.add_stream("h264")
    st.width = width
    st.height = height
    st.time_base = Fraction(1, 1000)  # milliseconds (MKV uses ms internally)

    extradata_set = False
    frame_duration_ms = 40  # 40ms per frame at 25fps

    for i, input_frame_num in enumerate(frames_to_encode):
        # Create and encode frame
        pixels = create_test_frame(width, height, input_frame_num)
        nv12 = bgra_to_nv12(pixels)
        force_idr = (i == 0)
        bitstream = encoder.encode(nv12, force_idr=force_idr)

        # Parse NALs to get SPS/PPS for extradata
        nals = parse_annexb(bitstream)

        if not extradata_set:
            sps_data = None
            pps_data = None
            for nal in nals:
                if nal.nal_unit_type == 7:
                    sps_data = nal.to_bytes()
                elif nal.nal_unit_type == 8:
                    pps_data = nal.to_bytes()
            if sps_data and pps_data:
                st.codec_context.extradata = (
                    b'\x00\x00\x00\x01' + sps_data +
                    b'\x00\x00\x00\x01' + pps_data
                )
                extradata_set = True
                print(f"  Set extradata: SPS={len(sps_data)}b, PPS={len(pps_data)}b")

        # Build frame data (ALL NALs, Annex B format)
        # PyAV expects complete AU including AU delimiter, SPS, PPS, and slices
        frame_data = bitstream  # Use the raw bitstream directly

        if not frame_data:
            print(f"  Frame {i}: no slice data, skipping")
            continue

        # Calculate PTS based on input frame number (VFR) - in milliseconds
        pts_ms = input_frame_num * frame_duration_ms

        # Create packet
        pkt = av.Packet(frame_data)
        pkt.stream = st
        pkt.pts = pts_ms
        pkt.dts = pts_ms
        pkt.time_base = st.time_base

        # Set keyframe flag
        is_idr = any(n.nal_unit_type == 5 for n in nals)
        if is_idr:
            pkt.is_keyframe = True

        out.mux(pkt)
        print(f"  Frame {i}: input={input_frame_num}, pts={pts_ms}ms, size={len(frame_data)}, idr={is_idr}")

    out.close()
    encoder.close()

    print(f"\nOutput: {mkv_path}")

    # Verify
    import subprocess
    print("\nVerifying with ffprobe...")
    result = subprocess.run([
        "ffprobe", "-v", "warning", str(mkv_path)
    ], capture_output=True, text=True)
    if result.stderr:
        print(f"Warnings:\n{result.stderr}")
    else:
        print("No errors!")

    # Check duration
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1", str(mkv_path)
    ], capture_output=True, text=True)
    print(f"Duration: {result.stdout.strip()}")

    print(f"\nTest with: ffplay {mkv_path}")


if __name__ == "__main__":
    main()
