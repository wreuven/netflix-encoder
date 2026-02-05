#!/usr/bin/env python3
"""
POC 5: P_Skip Frame Generation

Generates valid H.264 P-frames containing only P_Skip macroblocks.
Tests bitstream generation by creating I-frame + P_Skip sequence.

Usage:
    python3 poc5_pskip_generator.py

Output:
    - output/pskip_test.h264 — test sequence (I + P_Skip frames)
    - Validates with ffprobe/ffplay
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


class BitWriter:
    """Bitstream writer for H.264 NAL unit construction."""

    def __init__(self):
        self.data = bytearray()
        self.byte = 0
        self.bit_pos = 7  # MSB first

    def write_bit(self, bit):
        """Write a single bit."""
        if bit:
            self.byte |= (1 << self.bit_pos)
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.data.append(self.byte)
            self.byte = 0
            self.bit_pos = 7

    def write_bits(self, value, num_bits):
        """Write multiple bits (MSB first)."""
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_ue(self, value):
        """Write unsigned Exp-Golomb coded value."""
        # ue(v): 0 -> 1, 1 -> 010, 2 -> 011, 3 -> 00100, etc.
        value += 1
        num_bits = value.bit_length()
        # Write (num_bits - 1) zeros
        for _ in range(num_bits - 1):
            self.write_bit(0)
        # Write the value
        self.write_bits(value, num_bits)

    def write_se(self, value):
        """Write signed Exp-Golomb coded value."""
        # se(v): 0 -> 0, 1 -> 1, -1 -> 2, 2 -> 3, -2 -> 4, etc.
        if value <= 0:
            self.write_ue(-2 * value)
        else:
            self.write_ue(2 * value - 1)

    def byte_align(self):
        """Pad with zeros to byte boundary."""
        while self.bit_pos != 7:
            self.write_bit(0)

    def write_rbsp_trailing_bits(self):
        """Write RBSP trailing bits (1 followed by zeros to byte align)."""
        self.write_bit(1)
        self.byte_align()

    def to_bytes(self):
        """Return the bitstream as bytes, flushing any partial byte."""
        result = bytes(self.data)
        if self.bit_pos != 7:
            result += bytes([self.byte])
        return result


def add_emulation_prevention(data):
    """Add emulation prevention bytes (0x03) where needed."""
    result = bytearray()
    zero_count = 0

    for byte in data:
        if zero_count >= 2 and byte <= 0x03:
            result.append(0x03)  # Emulation prevention byte
            zero_count = 0

        result.append(byte)

        if byte == 0x00:
            zero_count += 1
        else:
            zero_count = 0

    return bytes(result)


def make_nal_unit(nal_ref_idc, nal_unit_type, rbsp_data):
    """Create a complete NAL unit with start code."""
    # Start code
    nal = bytearray([0x00, 0x00, 0x00, 0x01])

    # NAL header: forbidden_zero_bit (1) + nal_ref_idc (2) + nal_unit_type (5)
    nal_header = ((nal_ref_idc & 0x03) << 5) | (nal_unit_type & 0x1f)
    nal.append(nal_header)

    # Add emulation prevention to RBSP
    nal.extend(add_emulation_prevention(rbsp_data))

    return bytes(nal)


def generate_sps(width, height, profile_idc=66, level_idc=30):
    """Generate Sequence Parameter Set for Baseline profile."""
    width_mbs = (width + 15) // 16
    height_mbs = (height + 15) // 16

    # Crop if not MB-aligned
    crop_right = (width_mbs * 16 - width) // 2
    crop_bottom = (height_mbs * 16 - height) // 2
    frame_cropping = crop_right > 0 or crop_bottom > 0

    bits = BitWriter()

    # profile_idc
    bits.write_bits(profile_idc, 8)

    # constraint_set flags + reserved
    bits.write_bits(0, 8)  # constraint_set0-5 + reserved

    # level_idc
    bits.write_bits(level_idc, 8)

    # seq_parameter_set_id
    bits.write_ue(0)

    # log2_max_frame_num_minus4
    bits.write_ue(0)  # max_frame_num = 16

    # pic_order_cnt_type
    bits.write_ue(2)  # Type 2: no POC syntax in slice header

    # max_num_ref_frames
    bits.write_ue(1)

    # gaps_in_frame_num_value_allowed_flag
    bits.write_bit(0)

    # pic_width_in_mbs_minus1
    bits.write_ue(width_mbs - 1)

    # pic_height_in_map_units_minus1
    bits.write_ue(height_mbs - 1)

    # frame_mbs_only_flag (1 = frames only, no fields)
    bits.write_bit(1)

    # direct_8x8_inference_flag
    bits.write_bit(1)

    # frame_cropping_flag
    bits.write_bit(1 if frame_cropping else 0)
    if frame_cropping:
        bits.write_ue(0)           # left
        bits.write_ue(crop_right)  # right
        bits.write_ue(0)           # top
        bits.write_ue(crop_bottom) # bottom

    # vui_parameters_present_flag
    bits.write_bit(0)

    bits.write_rbsp_trailing_bits()

    return make_nal_unit(3, 7, bits.to_bytes())  # nal_ref_idc=3, nal_unit_type=7 (SPS)


def generate_pps():
    """Generate Picture Parameter Set."""
    bits = BitWriter()

    # pic_parameter_set_id
    bits.write_ue(0)

    # seq_parameter_set_id
    bits.write_ue(0)

    # entropy_coding_mode_flag (0 = CAVLC for Baseline)
    bits.write_bit(0)

    # bottom_field_pic_order_in_frame_present_flag
    bits.write_bit(0)

    # num_slice_groups_minus1
    bits.write_ue(0)

    # num_ref_idx_l0_default_active_minus1
    bits.write_ue(0)

    # num_ref_idx_l1_default_active_minus1
    bits.write_ue(0)

    # weighted_pred_flag
    bits.write_bit(0)

    # weighted_bipred_idc
    bits.write_bits(0, 2)

    # pic_init_qp_minus26
    bits.write_se(0)

    # pic_init_qs_minus26
    bits.write_se(0)

    # chroma_qp_index_offset
    bits.write_se(0)

    # deblocking_filter_control_present_flag
    bits.write_bit(1)

    # constrained_intra_pred_flag
    bits.write_bit(0)

    # redundant_pic_cnt_present_flag
    bits.write_bit(0)

    bits.write_rbsp_trailing_bits()

    return make_nal_unit(3, 8, bits.to_bytes())  # nal_ref_idc=3, nal_unit_type=8 (PPS)


def generate_pskip_slice(first_mb, mb_count, frame_num, width_mbs):
    """Generate a P slice containing only P_Skip macroblocks."""
    bits = BitWriter()

    # first_mb_in_slice
    bits.write_ue(first_mb)

    # slice_type (0 = P slice)
    bits.write_ue(0)

    # pic_parameter_set_id
    bits.write_ue(0)

    # frame_num (4 bits since log2_max_frame_num_minus4=0 means 4 bits)
    bits.write_bits(frame_num & 0xf, 4)

    # num_ref_idx_active_override_flag
    bits.write_bit(0)

    # ref_pic_list_modification_flag_l0
    bits.write_bit(0)

    # dec_ref_pic_marking: adaptive_ref_pic_marking_mode_flag
    bits.write_bit(0)

    # slice_qp_delta
    bits.write_se(0)

    # deblocking_filter_control_present_flag was set, so:
    # disable_deblocking_filter_idc
    bits.write_ue(1)  # 1 = disable deblocking

    # Slice data: mb_skip_run covers all MBs
    bits.write_ue(mb_count)

    bits.write_rbsp_trailing_bits()

    return make_nal_unit(2, 1, bits.to_bytes())  # nal_ref_idc=2, nal_unit_type=1 (non-IDR slice)


def generate_pskip_frame(width_mbs, height_mbs, frame_num):
    """Generate a complete P-frame with all P_Skip blocks."""
    frame_data = bytearray()

    # One slice per row for compatibility with splice operations
    for row in range(height_mbs):
        first_mb = row * width_mbs
        slice_data = generate_pskip_slice(first_mb, width_mbs, frame_num, width_mbs)
        frame_data.extend(slice_data)

    return bytes(frame_data)


def create_test_sequence(width, height, num_pskip_frames=30, output_path="output/pskip_test.h264"):
    """
    Create a test H.264 sequence:
    1. Generate SPS/PPS
    2. Encode one I-frame using FFmpeg (from gray image)
    3. Append P_Skip frames
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width_mbs = (width + 15) // 16
    height_mbs = (height + 15) // 16

    print(f"[poc5] Creating test sequence: {width}x{height} ({width_mbs}x{height_mbs} MBs)")
    print(f"[poc5] I-frame + {num_pskip_frames} P_Skip frames")

    # Generate SPS/PPS
    sps = generate_sps(width, height)
    pps = generate_pps()

    print(f"[poc5] SPS: {len(sps)} bytes")
    print(f"[poc5] PPS: {len(pps)} bytes")

    # Create a gray test image
    gray_frame = np.full((height, width, 3), 128, dtype=np.uint8)

    # Encode I-frame using FFmpeg (just one frame)
    iframe_path = output_path.parent / "temp_iframe.h264"

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
        "-level:v", "3.0",
        "-pix_fmt", "yuv420p",  # Required for baseline profile
        "-bf", "0",       # No B-frames
        "-refs", "1",     # Single reference
        "-g", "1",        # All I-frames
        "-f", "h264",
        str(iframe_path),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=gray_frame.tobytes())

    if proc.returncode != 0:
        print(f"[poc5] FFmpeg error: {stderr.decode()[-500:]}")
        return False

    # Read the I-frame (includes SPS/PPS from FFmpeg)
    with open(iframe_path, "rb") as f:
        iframe_data = f.read()

    print(f"[poc5] I-frame from FFmpeg: {len(iframe_data)} bytes")

    # Build complete output
    with open(output_path, "wb") as f:
        # Write I-frame (with FFmpeg's SPS/PPS)
        f.write(iframe_data)

        # Write P_Skip frames
        for i in range(num_pskip_frames):
            frame_num = (i + 1) % 16  # Wrap at 16 (log2_max_frame_num=4)
            pskip_frame = generate_pskip_frame(width_mbs, height_mbs, frame_num)
            f.write(pskip_frame)

    # Clean up temp file
    iframe_path.unlink()

    total_size = output_path.stat().st_size
    print(f"[poc5] Output: {output_path} ({total_size} bytes)")

    return True


def validate_output(h264_path):
    """Validate the H.264 file with ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames",
         "-of", "default=noprint_wrappers=1", str(h264_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[poc5] Validation failed: {result.stderr}")
        return False

    print(f"[poc5] ffprobe output:\n{result.stdout}")
    return True


def main():
    output_dir = Path(__file__).parent / "output"
    output_path = output_dir / "pskip_test.h264"

    # Use 640x360 for testing (common video size, MB-aligned)
    width, height = 640, 360

    print("=" * 60)
    print("POC 5: P_Skip Frame Generation")
    print("=" * 60)
    print()

    success = create_test_sequence(
        width=width,
        height=height,
        num_pskip_frames=30,
        output_path=str(output_path),
    )

    if not success:
        print("[poc5] Failed to create test sequence")
        return 1

    print()
    print("Validating output...")
    if validate_output(output_path):
        print("[poc5] Validation passed!")
    else:
        print("[poc5] Validation failed")
        return 1

    print()
    print("Test playback:")
    print(f"  ffplay {output_path}")
    print(f"  python3 ../h264-newpoc/splicer/staging_player.py {output_path}")
    print()

    # Also create MP4 version for easier playback
    mp4_path = output_dir / "pskip_test.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(output_path),
        "-c:v", "copy", str(mp4_path)
    ], capture_output=True)

    if mp4_path.exists():
        print(f"  MP4 version: {mp4_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
