#!/usr/bin/env python3
"""
POC: VFR MKV output test

Creates a simple VFR video by encoding frames with varying timestamps.
This tests the mechanism for skipping frames (VFR) without the complexity
of the full encoding pipeline.
"""

import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

# Add splicer path
sys.path.insert(0, "/home/wachtfogel/h264-newpoc/splicer")
from nvenc_encoder import NVENCEncoder
from h264_splicer import parse_annexb

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_test_frame(width, height, frame_num):
    """Create a simple test frame with frame number visible."""
    # Create gradient background
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Gradient based on frame number
    color = (frame_num * 10) % 256
    frame[:, :, 0] = color  # B
    frame[:, :, 1] = (255 - color)  # G
    frame[:, :, 2] = 128  # R
    frame[:, :, 3] = 255  # A

    # Add a moving box to show motion
    box_x = (frame_num * 20) % (width - 100)
    box_y = 100
    frame[box_y:box_y+100, box_x:box_x+100, :3] = 255

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

    # Create encoder
    encoder = NVENCEncoder(width, height, qp=23)

    # Output paths
    h264_path = OUTPUT_DIR / "vfr_test.h264"
    mkv_path = OUTPUT_DIR / "vfr_test.mkv"

    # Track frames and their timestamps
    # Simulate: encode frames 0, 1, 2, skip 3, 4, encode 5, 6, skip 7, encode 8, 9
    # This creates VFR: frame at 0ms, 40ms, 80ms, 200ms (skip 3,4), 240ms, ...

    frames_to_encode = [0, 1, 2, 5, 6, 8, 9, 10, 11, 12]  # Skip frames 3, 4, 7

    print(f"Encoding {len(frames_to_encode)} frames with VFR directly to MKV...")

    # Open MKV output
    output = av.open(str(mkv_path), 'w')
    stream = output.add_stream('h264')
    stream.width = width
    stream.height = height
    stream.time_base = Fraction(1, 1000)  # milliseconds

    extradata_set = False
    h264_file = open(h264_path, "wb")  # Also write raw for debugging

    for i, input_frame_num in enumerate(frames_to_encode):
        # Create test frame
        pixels = create_test_frame(width, height, input_frame_num)
        nv12 = bgra_to_nv12(pixels)

        # Encode
        force_idr = (i == 0)
        bitstream = encoder.encode(nv12, force_idr=force_idr)

        # Write raw H.264 for debugging
        h264_file.write(bitstream)

        # Parse NALs to extract SPS/PPS for extradata
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
                stream.codec_context.extradata = (
                    b'\x00\x00\x00\x01' + sps_data +
                    b'\x00\x00\x00\x01' + pps_data
                )
                extradata_set = True

        # Build frame data (only slice NALs, not SPS/PPS)
        frame_data = b''
        for nal in nals:
            if nal.nal_unit_type in (1, 5):  # Slice NALs only
                frame_data += b'\x00\x00\x00\x01' + nal.to_bytes()

        if not frame_data:
            continue

        # Create packet with VFR timestamp
        packet = av.Packet(frame_data)
        pts_ms = input_frame_num * 40  # 40ms per input frame at 25fps
        packet.pts = pts_ms
        packet.dts = pts_ms
        packet.stream = stream

        # Set keyframe
        is_idr = any(n.nal_unit_type == 5 for n in nals)
        packet.is_keyframe = is_idr

        output.mux(packet)

        print(f"  Frame {i}: input={input_frame_num}, pts={pts_ms}ms, idr={is_idr}")

    h264_file.close()
    output.close()
    encoder.close()

    print(f"\nRaw H.264: {h264_path}")
    print(f"VFR MKV (PyAV - may be broken): {mkv_path}")

    # Verify the H.264 is valid
    print("\nVerifying H.264...")
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_name,width,height",
        "-of", "default=noprint_wrappers=1",
        str(h264_path)
    ], capture_output=True, text=True)
    print(result.stdout)

    # Create VFR MKV using ffmpeg with concat demuxer
    print("\nCreating VFR MKV via ffmpeg concat...")

    # Create concat file with durations
    concat_path = OUTPUT_DIR / "vfr_concat.txt"
    with open(concat_path, "w") as f:
        # ffmpeg concat demuxer format
        f.write("ffconcat version 1.0\n")
        # Write raw H264 as single file, then use setpts filter for VFR
        f.write(f"file '{h264_path.name}'\n")

    # Actually, concat demuxer doesn't help with VFR for a single file
    # Let's use a different approach: write individual frame files

    # Better approach: use -vsync vfr with setpts filter
    # But we need to know frame timestamps at decode time

    # Simplest VFR approach with ffmpeg: use -itsoffset per frame
    # Not practical for many frames

    # Best approach: write timecode file and use mkvmerge (requires install)
    # Or: decode, re-encode with timestamps (loses quality)
    # Or: use ffmpeg -vf setpts with expression

    # For now, just copy as CFR and note the limitation
    mkv_ffmpeg_path = OUTPUT_DIR / "vfr_test_ffmpeg.mkv"
    result = subprocess.run([
        "ffmpeg", "-y", "-r", "25",
        "-i", str(h264_path),
        "-c:v", "copy",
        str(mkv_ffmpeg_path)
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"MKV (ffmpeg CFR): {mkv_ffmpeg_path}")
    else:
        print(f"ffmpeg failed: {result.stderr}")

    # Verify ffmpeg MKV
    print("\nVerifying ffmpeg MKV...")
    result = subprocess.run([
        "ffprobe", "-v", "warning",
        str(mkv_ffmpeg_path)
    ], capture_output=True, text=True)
    print(result.stderr if result.stderr else "No errors!")
    print(f"\nTest with: ffplay {mkv_ffmpeg_path}")


if __name__ == "__main__":
    main()
