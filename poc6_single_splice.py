#!/usr/bin/env python3
"""
POC 6: Single-Region Splice

Encodes a region separately and splices its slices into a P_Skip frame.
This demonstrates the core technique from LEARNINGS.md:
1. Encode background as I-frame
2. Encode region with per-row slices (with BG padding)
3. Transplant region slices into P_Skip frame

Usage:
    python3 poc6_single_splice.py

Output:
    - output/spliced_single.h264 — spliced stream
    - output/spliced_single.mp4 — MP4 version
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

# Add h264-newpoc splicer to path
SPLICER_PATH = Path("/home/wachtfogel/h264-newpoc/splicer")
sys.path.insert(0, str(SPLICER_PATH))

from h264_splicer import (
    parse_annexb, NALUnit, BitWriter, BitReader, SPS, PPS,
    rewrite_slice_header, add_emulation_prevention,
)


def create_pskip_slice(first_mb, mb_count, frame_num, sps, nal_ref_idc=2):
    """Create a P-slice with all P_Skip macroblocks."""
    w = BitWriter()

    # Slice header
    w.write_ue(first_mb)   # first_mb_in_slice
    w.write_ue(0)          # slice_type = P
    w.write_ue(0)          # pic_parameter_set_id

    # frame_num (use SPS to get correct bit width)
    frame_num_bits = sps.log2_max_frame_num_minus4 + 4
    w.write_bits(frame_num % (1 << frame_num_bits), frame_num_bits)

    # num_ref_idx_active_override_flag = 0
    w.write_bits(0, 1)

    # ref_pic_list_modification_flag_l0 = 0
    w.write_bits(0, 1)

    # dec_ref_pic_marking (if nal_ref_idc > 0)
    if nal_ref_idc > 0:
        w.write_bits(0, 1)  # adaptive_ref_pic_marking_mode_flag = 0

    # slice_qp_delta = 0
    w.write_se(0)

    # disable_deblocking_filter_idc = 1 (disable)
    w.write_ue(1)

    # Slice data: mb_skip_run covers all MBs
    w.write_ue(mb_count)

    rbsp = w.finish_rbsp()
    return NALUnit(nal_ref_idc, 1, rbsp)  # nal_unit_type=1 (non-IDR)


def encode_region_with_bg_padding(region_pixels, bg_pixels, region_x, region_y,
                                   output_path, padding_mbs=1):
    """
    Encode a region with BG padding on all sides.

    Args:
        region_pixels: Region content (H, W, 3) RGB
        bg_pixels: Full background frame (H, W, 3) RGB
        region_x, region_y: Position of region in full frame
        output_path: Where to write H.264
        padding_mbs: Number of MB rows/cols of BG padding

    Returns:
        List of NALUnit objects (slices), or None on failure
    """
    region_h, region_w = region_pixels.shape[:2]
    padding_px = padding_mbs * 16

    # Calculate padded region bounds
    padded_x = max(0, region_x - padding_px)
    padded_y = max(0, region_y - padding_px)
    padded_w = region_w + 2 * padding_px
    padded_h = region_h + 2 * padding_px

    # Clamp to frame bounds
    frame_h, frame_w = bg_pixels.shape[:2]
    padded_x2 = min(frame_w, padded_x + padded_w)
    padded_y2 = min(frame_h, padded_y + padded_h)
    padded_w = padded_x2 - padded_x
    padded_h = padded_y2 - padded_y

    # Create padded composite: BG crop with region overlaid
    composite = bg_pixels[padded_y:padded_y2, padded_x:padded_x2].copy()

    # Overlay region content at correct position within composite
    ov_x = region_x - padded_x
    ov_y = region_y - padded_y
    composite[ov_y:ov_y+region_h, ov_x:ov_x+region_w] = region_pixels

    # Number of MB rows for slices
    height_mbs = padded_h // 16
    width_mbs = padded_w // 16

    # Encode with per-row slices using x264 constraints
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{padded_w}x{padded_h}",
        "-r", "30",
        "-i", "pipe:",
        "-frames:v", "1",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-pix_fmt", "yuv420p",
        # Critical constraints from LEARNINGS.md
        "-x264opts", f"slice-max-mbs={width_mbs}:no-deblock:ref=1:bframes=0:subme=2",
        "-f", "h264",
        str(output_path),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=composite.tobytes())

    if proc.returncode != 0:
        print(f"[poc6] Region encode failed: {stderr.decode()[-500:]}")
        return None

    # Parse encoded region
    with open(output_path, "rb") as f:
        data = f.read()

    nals = parse_annexb(data)

    # Extract slices (IDR or non-IDR)
    slices = [n for n in nals if n.nal_unit_type in (1, 5)]

    return slices, nals


def create_spliced_sequence(width, height, region_x, region_y, region_w, region_h,
                            num_frames=30, output_path="output/spliced_single.h264"):
    """
    Create a spliced H.264 sequence demonstrating region injection.

    1. First frame: Full I-frame (gray background)
    2. Subsequent frames: P_Skip everywhere + transplanted region slices
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width_mbs = (width + 15) // 16
    height_mbs = (height + 15) // 16

    # Region position in MBs
    region_mb_x = region_x // 16
    region_mb_y = region_y // 16
    region_mb_w = region_w // 16
    region_mb_h = region_h // 16

    # Padding (1 MB on all sides)
    padding_mbs = 1
    padded_mb_x = region_mb_x - padding_mbs
    padded_mb_y = region_mb_y - padding_mbs
    padded_mb_w = region_mb_w + 2 * padding_mbs
    padded_mb_h = region_mb_h + 2 * padding_mbs

    print(f"[poc6] Frame: {width}x{height} ({width_mbs}x{height_mbs} MBs)")
    print(f"[poc6] Region: ({region_x},{region_y}) {region_w}x{region_h} "
          f"(MB: [{region_mb_x},{region_mb_y}] {region_mb_w}x{region_mb_h})")
    print(f"[poc6] Padded region: MB [{padded_mb_x},{padded_mb_y}] {padded_mb_w}x{padded_mb_h}")

    # Background: gray
    bg_frame = np.full((height, width, 3), 128, dtype=np.uint8)

    # Region content generator
    def make_region_content(frame_num):
        """Create colored region that changes each frame."""
        region = np.zeros((region_h, region_w, 3), dtype=np.uint8)
        # Color cycling
        r = (frame_num * 8) % 256
        g = (128 + frame_num * 4) % 256
        b = (255 - frame_num * 8) % 256
        region[:, :] = [r, g, b]
        # Add stripe indicator
        stripe_y = (frame_num * 10) % region_h
        region[stripe_y:min(stripe_y+5, region_h), :] = [255, 255, 255]
        return region

    # Encode background I-frame
    bg_path = output_path.parent / "temp_bg.h264"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", "30",
        "-i", "pipe:",
        "-frames:v", "1",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-pix_fmt", "yuv420p",
        "-x264opts", f"slice-max-mbs={width_mbs}:no-deblock:ref=1:bframes=0",
        "-f", "h264",
        str(bg_path),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=bg_frame.tobytes())

    if proc.returncode != 0:
        print(f"[poc6] BG encode failed: {stderr.decode()[-500:]}")
        return False

    # Parse background to get SPS/PPS
    with open(bg_path, "rb") as f:
        bg_data = f.read()

    bg_nals = parse_annexb(bg_data)
    bg_sps_nal = next((n for n in bg_nals if n.nal_unit_type == 7), None)
    bg_pps_nal = next((n for n in bg_nals if n.nal_unit_type == 8), None)

    if not bg_sps_nal or not bg_pps_nal:
        print("[poc6] Failed to find SPS/PPS in background")
        return False

    bg_sps = SPS.from_rbsp(bg_sps_nal.rbsp)
    bg_pps = PPS.from_rbsp(bg_pps_nal.rbsp)

    print(f"[poc6] Background I-frame: {len(bg_data)} bytes")
    print(f"[poc6] SPS: log2_max_frame_num_minus4={bg_sps.log2_max_frame_num_minus4}")

    # Build output
    output_nals = []

    # Add SPS/PPS and background IDR
    for nal in bg_nals:
        output_nals.append(nal)

    # Process subsequent frames
    region_path = output_path.parent / "temp_region.h264"

    for frame_idx in range(1, num_frames):
        frame_num = frame_idx % (1 << (bg_sps.log2_max_frame_num_minus4 + 4))

        # Create region content
        region_pixels = make_region_content(frame_idx)

        # Encode region with BG padding
        result = encode_region_with_bg_padding(
            region_pixels, bg_frame,
            region_x, region_y,
            region_path,
            padding_mbs=padding_mbs
        )

        if result is None:
            print(f"[poc6] Warning: region encode failed for frame {frame_idx}")
            # Fall back to all P_Skip
            for row in range(height_mbs):
                first_mb = row * width_mbs
                output_nals.append(create_pskip_slice(first_mb, width_mbs, frame_num, bg_sps))
            continue

        region_slices, region_all_nals = result

        # Get region SPS/PPS for rewriting
        region_sps_nal = next((n for n in region_all_nals if n.nal_unit_type == 7), None)
        region_pps_nal = next((n for n in region_all_nals if n.nal_unit_type == 8), None)

        if not region_sps_nal or not region_pps_nal:
            print(f"[poc6] Warning: no SPS/PPS in region for frame {frame_idx}")
            for row in range(height_mbs):
                first_mb = row * width_mbs
                output_nals.append(create_pskip_slice(first_mb, width_mbs, frame_num, bg_sps))
            continue

        region_sps = SPS.from_rbsp(region_sps_nal.rbsp)
        region_pps = PPS.from_rbsp(region_pps_nal.rbsp)

        # Group region slices by row
        region_rows = {}
        for nal in region_slices:
            reader = BitReader(nal.rbsp)
            first_mb = reader.read_ue()
            row = first_mb // padded_mb_w
            region_rows[row] = nal

        # Build output frame
        for row in range(height_mbs):
            first_mb = row * width_mbs

            # Check if this row intersects with padded region
            if padded_mb_y <= row < padded_mb_y + padded_mb_h:
                region_local_row = row - padded_mb_y

                # Left P_Skip (before padded region)
                if padded_mb_x > 0:
                    output_nals.append(create_pskip_slice(
                        first_mb, padded_mb_x, frame_num, bg_sps
                    ))

                # Transplanted region slice
                if region_local_row in region_rows:
                    ov_nal = region_rows[region_local_row]
                    new_first_mb = first_mb + padded_mb_x

                    # Rewrite slice header using h264_splicer
                    is_idr = ov_nal.nal_unit_type == 5
                    rewritten = rewrite_slice_header(
                        ov_nal,
                        region_sps,
                        region_pps,
                        new_first_mb=new_first_mb,
                        new_idc=1,  # disable deblocking
                        new_frame_num=frame_num,
                        convert_idr_to_non_idr=is_idr,
                        new_nal_ref_idc=2,
                        target_sps=bg_sps
                    )
                    output_nals.append(rewritten)
                else:
                    # Fallback to P_Skip if slice missing
                    output_nals.append(create_pskip_slice(
                        first_mb + padded_mb_x, padded_mb_w, frame_num, bg_sps
                    ))

                # Right P_Skip (after padded region)
                right_start = padded_mb_x + padded_mb_w
                if right_start < width_mbs:
                    output_nals.append(create_pskip_slice(
                        first_mb + right_start, width_mbs - right_start, frame_num, bg_sps
                    ))
            else:
                # Full row P_Skip
                output_nals.append(create_pskip_slice(first_mb, width_mbs, frame_num, bg_sps))

        if frame_idx == 1:
            print(f"[poc6] Frame 1: transplanted {len(region_rows)} region slices")

    # Write output
    with open(output_path, "wb") as f:
        for nal in output_nals:
            f.write(b'\x00\x00\x00\x01')
            f.write(nal.to_bytes())

    # Clean up
    bg_path.unlink()
    if region_path.exists():
        region_path.unlink()

    total_size = output_path.stat().st_size
    print(f"[poc6] Output: {output_path} ({total_size} bytes)")

    return True


def validate_and_convert(h264_path):
    """Validate and convert to MP4."""
    # Validate
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames",
         "-of", "default=nokey=1:noprint_wrappers=1",
         str(h264_path)],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        frames = result.stdout.strip()
        print(f"[poc6] Decoded {frames} frames")
    else:
        print(f"[poc6] Validation warning: {result.stderr}")

    # Convert to MP4
    mp4_path = h264_path.with_suffix(".mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(h264_path),
        "-c:v", "copy", str(mp4_path)
    ], capture_output=True)

    if mp4_path.exists():
        print(f"[poc6] MP4: {mp4_path}")

    return True


def main():
    output_dir = Path(__file__).parent / "output"
    output_path = output_dir / "spliced_single.h264"

    # Test parameters
    width, height = 640, 360  # MB-aligned
    # Region (MB-aligned, with room for 1MB padding on all sides)
    region_x, region_y = 176, 96   # 11 MBs from left, 6 from top (allows 1MB padding)
    region_w, region_h = 128, 96   # 8x6 MBs

    print("=" * 60)
    print("POC 6: Single-Region Splice")
    print("=" * 60)
    print()

    success = create_spliced_sequence(
        width=width,
        height=height,
        region_x=region_x,
        region_y=region_y,
        region_w=region_w,
        region_h=region_h,
        num_frames=30,
        output_path=str(output_path),
    )

    if not success:
        print("[poc6] Failed to create sequence")
        return 1

    print()
    validate_and_convert(output_path)

    print()
    print("Test playback:")
    print(f"  ffplay {output_path}")
    print(f"  python3 ../h264-newpoc/splicer/staging_player.py {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
