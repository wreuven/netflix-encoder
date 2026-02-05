#!/usr/bin/env python3
"""
POC 7: Live Integration with Two-Encoder Architecture

Two encoder contexts using NVENC for proper I/P frame production:
1. Full-Frame Encoder (Context 1): For "changed" frames
   - Persistent NVENC encoder
   - Stores output as LTR 0 (via force_idr when resuming after stitched frames)
   - Produces I-frame first, then P-frames for consecutive changed frames

2. Region/Overlay Encoder (Context 2): For "video_only" frames
   - Per-row slices, no-deblock
   - Output slices transplanted into stitched frames

Stitched frames (programmatically generated):
- For "unchanged": full P_Skip frame (refs prev)
- For "video_only": P_Skip + transplanted region slices (refs prev)

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler poc7_live_splice_handler.py
"""

import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Add paths
TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
SPLICER_PATH = Path("/home/wachtfogel/h264-newpoc/splicer")
sys.path.insert(0, str(TRACER_PATH))
sys.path.insert(0, str(SPLICER_PATH))

from frame_handler import BaseFrameHandler
from h264_splicer import (
    parse_annexb, NALUnit, BitWriter, BitReader, SPS, PPS,
    rewrite_slice_header, add_emulation_prevention, parse_slice_header,
    modify_sps_num_ref_frames,
)
from nvenc_encoder import NVENCEncoder


def bgra_to_nv12(bgra):
    """Convert BGRA numpy array to NV12 format for NVENC."""
    # Extract BGR channels and convert to RGB
    b = bgra[:, :, 0].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    r = bgra[:, :, 2].astype(np.float32)

    # RGB to YUV conversion (BT.601)
    y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
    u = ((-0.169 * r - 0.331 * g + 0.500 * b) + 128).clip(0, 255).astype(np.uint8)
    v = ((0.500 * r - 0.419 * g - 0.081 * b) + 128).clip(0, 255).astype(np.uint8)

    height, width = y.shape

    # Subsample U and V (4:2:0)
    u_sub = u[::2, ::2]
    v_sub = v[::2, ::2]

    # Interleave U and V for NV12 format
    uv = np.zeros((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u_sub
    uv[:, 1::2] = v_sub

    # Stack Y and UV planes
    nv12 = np.vstack([y, uv])
    return nv12


def create_pskip_slice(first_mb, mb_count, frame_num, sps, nal_ref_idc=2, ref_ltr0=False):
    """Create a P-slice with all P_Skip macroblocks.

    Args:
        ref_ltr0: If True, explicitly reference LTR 0 instead of default reference.
    """
    w = BitWriter()

    w.write_ue(first_mb)
    w.write_ue(0)  # slice_type = P
    w.write_ue(0)  # pic_parameter_set_id

    frame_num_bits = sps.log2_max_frame_num_minus4 + 4
    w.write_bits(frame_num % (1 << frame_num_bits), frame_num_bits)

    w.write_bits(0, 1)  # num_ref_idx_active_override_flag

    if ref_ltr0:
        # Explicitly reference LTR 0
        w.write_bits(1, 1)  # ref_pic_list_modification_flag_l0 = 1
        w.write_ue(2)       # modification_of_pic_nums_idc = 2 (long-term ref)
        w.write_ue(0)       # long_term_pic_num = 0 (LTR 0)
        w.write_ue(3)       # modification_of_pic_nums_idc = 3 (end)
    else:
        w.write_bits(0, 1)  # ref_pic_list_modification_flag_l0 = 0

    if nal_ref_idc > 0:
        w.write_bits(0, 1)  # adaptive_ref_pic_marking_mode_flag

    w.write_se(0)  # slice_qp_delta
    w.write_ue(1)  # disable_deblocking_filter_idc
    w.write_ue(mb_count)  # mb_skip_run

    rbsp = w.finish_rbsp()
    return NALUnit(nal_ref_idc, 1, rbsp)


class FullFrameEncoder:
    """
    Encoder Context 1: Full-frame encoder for "changed" frames.

    Uses persistent NVENC encoder with bitstream post-processing for LTR 0:
    - First frame is IDR, marked as LTR 0
    - All P-frames reference LTR 0 and are then marked as new LTR 0
    - This creates a chain: each frame refs previous LTR 0, becomes new LTR 0
    """

    def __init__(self, width, height, output_dir):
        self.width = width
        self.height = height
        self.output_dir = output_dir

        self.width_mbs = (width + 15) // 16
        self.height_mbs = (height + 15) // 16

        # Create persistent NVENC encoder
        # slice_mode=3 means total number of slices = height_mbs (one per row)
        self._encoder = NVENCEncoder(
            width, height,
            slice_mode=3,  # Total number of slices
            slice_mode_data=self.height_mbs,  # One slice per MB row
            disable_deblock=False,  # Full frame can use deblocking
            qp=23
        )

        # State tracking
        self.sps = None
        self.pps = None
        self.frame_num = 0
        self.stitched_frames_since_last_encode = 0
        self.encode_count = 0
        self.idr_count = 0
        self.p_count = 0
        self.ltr_ref_count = 0  # Count of frames that referenced LTR 0

    def notify_stitched_frame(self):
        """Called when a stitched frame is output instead of encoder output."""
        self.stitched_frames_since_last_encode += 1

    def _rewrite_slice_for_ltr0(self, nal, ref_ltr0=False, frame_num=None):
        """
        Rewrite slice header to:
        1. Mark this frame as LTR 0 (always)
        2. Reference LTR 0 instead of short-term ref (if ref_ltr0=True)
        3. Use correct frame_num (accounting for stitched frames)
        """
        is_idr = nal.nal_unit_type == 5

        # Parse original slice header to get values to preserve
        header, _ = parse_slice_header(nal, self.sps, self.pps)

        # Determine dec_ref_pic_marking for LTR 0
        if is_idr:
            # For IDR: set long_term_reference_flag=1
            marking = {'long_term': True}
        else:
            # For non-IDR: use MMCO 6 to mark as LTR 0
            # MMCO 6 = mark current picture as long-term with index 0
            marking = {'mmco': [(6, 0)]}

        # Determine ref_pic_list_modification
        if ref_ltr0 and not is_idr:
            # Reference LTR 0 instead of short-term reference
            ref_mod = {'type': 'long_term', 'idx': 0}
        else:
            ref_mod = None  # Keep default reference

        # Rewrite the slice header
        rewritten = rewrite_slice_header(
            nal, self.sps, self.pps,
            new_first_mb=header.first_mb_in_slice,
            new_idc=header.disable_deblocking_filter_idc,
            new_frame_num=frame_num,  # Use provided frame_num
            new_ref_pic_list_mod=ref_mod,
            new_dec_ref_pic_marking=marking,
        )

        return rewritten

    def encode(self, pixels, frame_num=None):
        """
        Encode full frame with LTR 0 post-processing.

        - Always marks output as LTR 0
        - When resuming after stitched frames: references LTR 0 (efficient P-frame)

        Args:
            pixels: BGRA pixel data
            frame_num: Frame number to use in bitstream (from frame_builder)
        Returns: list of NAL units
        """
        # Only force IDR for the very first frame
        force_idr = (self.sps is None)
        # Always reference LTR 0 for P-frames (since every frame is marked as LTR 0)
        need_ltr_ref = True

        # Convert BGRA to NV12
        nv12 = bgra_to_nv12(pixels)

        # Encode with NVENC
        bitstream = self._encoder.encode(nv12, force_idr=force_idr)

        # Parse NAL units
        nals = parse_annexb(bitstream)

        # Extract SPS/PPS
        output_nals = []
        for nal in nals:
            if nal.nal_unit_type == 7:
                self.sps = SPS.from_rbsp(nal.rbsp)
                output_nals.append(nal)
            elif nal.nal_unit_type == 8:
                self.pps = PPS.from_rbsp(nal.rbsp)
                output_nals.append(nal)
            elif nal.nal_unit_type in (1, 5):
                # Rewrite slice to mark as LTR 0 (and ref LTR 0 if needed)
                # Use provided frame_num to stay in sync with stitched frames
                rewritten = self._rewrite_slice_for_ltr0(nal, ref_ltr0=need_ltr_ref, frame_num=frame_num)
                output_nals.append(rewritten)
            else:
                output_nals.append(nal)

        # Track frame types
        has_idr = any(n.nal_unit_type == 5 for n in output_nals)
        if has_idr:
            self.idr_count += 1
        else:
            self.p_count += 1
            if need_ltr_ref:
                self.ltr_ref_count += 1

        self.stitched_frames_since_last_encode = 0
        self.encode_count += 1

        return output_nals

    def get_sps_pps(self):
        """Return current SPS/PPS for stitched frame generation."""
        return self.sps, self.pps

    def close(self):
        """Close the encoder."""
        if self._encoder:
            self._encoder.close()
            self._encoder = None


class RegionEncoder:
    """
    Encoder Context 2: Region encoder for "video_only" frames.

    Uses NVENC with:
    - Per-row slices for transplantation
    - No deblocking to prevent splice artifacts
    - Maintains encoder state for I/P frame production

    For P-frames to work when transplanted, the padding pixels must match
    what the decoder sees as the reference. We achieve this by:
    1. Storing the padding pixels when we produce an IDR
    2. Reusing those SAME padding pixels for subsequent P-frames
    This ensures encoder2's reference matches its input for the padding area.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.encode_count = 0
        self.idr_count = 0
        self.p_count = 0

        # Cache of encoders by region size (width, height) -> encoder
        self._encoders = {}
        # Cache of SPS/PPS by region size (width, height) -> (sps, pps)
        self._sps_pps_cache = {}
        self._last_region_size = None

        # Store padding pixels from IDR frame for reuse in P-frames
        # This ensures encoder2's reference padding matches input padding
        self._cached_padding = None  # (top, bottom, left, right) strips
        self._cached_padding_bounds = None  # (padded_x, padded_y, padded_w, padded_h, video_x, video_y, video_w, video_h)

        # Debug: write raw encoder2 output to file
        self._debug_file = open(output_dir / "encoder2_debug.h264", "wb")
        self._debug_sps_written = False

    def _get_encoder(self, width, height):
        """Get or create encoder for given region size."""
        key = (width, height)
        if key not in self._encoders:
            height_mbs = height // 16
            # Create new encoder with per-row slices
            self._encoders[key] = NVENCEncoder(
                width, height,
                slice_mode=2,  # MB rows per slice
                slice_mode_data=1,  # 1 MB row per slice
                disable_deblock=True,  # Critical for splicing
                qp=23
            )
        return self._encoders[key]

    def encode_region(self, pixels, region_x, region_y, region_w, region_h, force_idr=False):
        """
        Encode a region with 1 MB background padding.

        For P-frames to work correctly when transplanted, we reuse the padding
        pixels from the IDR frame. This ensures encoder2's reference padding
        matches the input padding, so motion vectors into the padding area
        reference the correct content.

        Returns: (slices, region_info) or (None, None) on failure
                 region_info = (sps, pps, mb_x, mb_y, mb_w, mb_h)
        """
        padding_mbs = 1
        padding_px = padding_mbs * 16

        # Calculate padded region bounds (with 1 MB border)
        frame_h, frame_w = pixels.shape[:2]

        padded_x = max(0, (region_x // 16) * 16 - padding_px)
        padded_y = max(0, (region_y // 16) * 16 - padding_px)
        padded_x2 = min(frame_w, ((region_x + region_w + 15) // 16) * 16 + padding_px)
        padded_y2 = min(frame_h, ((region_y + region_h + 15) // 16) * 16 + padding_px)

        padded_w = ((padded_x2 - padded_x) // 16) * 16
        padded_h = ((padded_y2 - padded_y) // 16) * 16

        if padded_w < 32 or padded_h < 32:
            return None, None

        # Video region within padded area (excluding padding)
        video_x_in_padded = (region_x // 16) * 16 - padded_x
        video_y_in_padded = (region_y // 16) * 16 - padded_y
        video_w_in_padded = ((region_x + region_w + 15) // 16) * 16 - (region_x // 16) * 16
        video_h_in_padded = ((region_y + region_h + 15) // 16) * 16 - (region_y // 16) * 16

        # Extract full padded region from current frame
        region_bgra = pixels[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w].copy()

        # Check if we need IDR
        current_size = (padded_w, padded_h)
        size_changed = (current_size != self._last_region_size)
        no_sps_pps = (current_size not in self._sps_pps_cache)
        need_idr = force_idr or size_changed or no_sps_pps

        # NOTE: Padding caching disabled for debugging position issue
        # if need_idr:
        #     # Store the padding pixels for reuse in subsequent P-frames
        #     # This ensures encoder2's reference matches input for P-frames
        #     self._cached_padding = region_bgra.copy()
        #     self._cached_padding_bounds = (padded_x, padded_y, padded_w, padded_h,
        #                                    video_x_in_padded, video_y_in_padded,
        #                                    video_w_in_padded, video_h_in_padded)
        # elif self._cached_padding is not None and self._cached_padding_bounds is not None:
        #     # For P-frames: use cached padding, but update the video region with current content
        #     cached_bounds = self._cached_padding_bounds
        #     if (padded_x == cached_bounds[0] and padded_y == cached_bounds[1] and
        #         padded_w == cached_bounds[2] and padded_h == cached_bounds[3]):
        #         # Same bounds - reuse cached padding for border, update video region
        #         vx, vy = cached_bounds[4], cached_bounds[5]
        #         vw, vh = cached_bounds[6], cached_bounds[7]
        #
        #         # Start with cached frame (has correct padding)
        #         region_bgra = self._cached_padding.copy()
        #         # Update only the video region with current pixels
        #         current_video = pixels[padded_y+vy:padded_y+vy+vh, padded_x+vx:padded_x+vx+vw]
        #         region_bgra[vy:vy+vh, vx:vx+vw] = current_video

        # Convert to NV12
        nv12 = bgra_to_nv12(region_bgra)

        self._last_region_size = current_size

        # Get encoder for this size
        encoder = self._get_encoder(padded_w, padded_h)

        # Encode
        bitstream = encoder.encode(nv12, force_idr=need_idr)

        # Debug: write raw encoder output to file
        self._debug_file.write(bitstream)

        # Parse NAL units
        nals = parse_annexb(bitstream)

        # Extract SPS/PPS from IDR frames and cache them
        for nal in nals:
            if nal.nal_unit_type == 7:
                sps = SPS.from_rbsp(nal.rbsp)
                if current_size not in self._sps_pps_cache:
                    self._sps_pps_cache[current_size] = (sps, None)
                else:
                    self._sps_pps_cache[current_size] = (sps, self._sps_pps_cache[current_size][1])
            elif nal.nal_unit_type == 8:
                pps = PPS.from_rbsp(nal.rbsp)
                if current_size not in self._sps_pps_cache:
                    self._sps_pps_cache[current_size] = (None, pps)
                else:
                    self._sps_pps_cache[current_size] = (self._sps_pps_cache[current_size][0], pps)

        # Get cached SPS/PPS for this size
        region_sps, region_pps = self._sps_pps_cache.get(current_size, (None, None))

        # Get slice NALs
        slices = [n for n in nals if n.nal_unit_type in (1, 5)]

        # Track frame types
        has_idr = any(n.nal_unit_type == 5 for n in nals)
        if has_idr:
            self.idr_count += 1
        else:
            self.p_count += 1

        self.encode_count += 1

        region_info = (
            region_sps, region_pps,
            padded_x // 16, padded_y // 16,  # MB position
            padded_w // 16, padded_h // 16   # MB dimensions
        )

        return slices, region_info

    def close(self):
        """Close all encoders."""
        for encoder in self._encoders.values():
            encoder.close()
        self._encoders.clear()
        self._sps_pps_cache.clear()
        self._cached_padding = None
        self._cached_padding_bounds = None
        if self._debug_file:
            self._debug_file.close()
            self._debug_file = None


class StitchedFrameBuilder:
    """Builds programmatically generated stitched frames."""

    def __init__(self, width_mbs, height_mbs):
        self.width_mbs = width_mbs
        self.height_mbs = height_mbs
        self.frame_num = 0
        self.sps = None

    def set_sps(self, sps):
        """Set the SPS to use for frame_num encoding."""
        self.sps = sps

    def increment_frame_num(self):
        """Increment frame_num for next stitched frame."""
        if self.sps:
            max_frame_num = 1 << (self.sps.log2_max_frame_num_minus4 + 4)
            self.frame_num = (self.frame_num + 1) % max_frame_num

    def reset_frame_num(self):
        """Reset frame_num (after IDR from Encoder 1)."""
        self.frame_num = 0

    def build_full_pskip_frame(self):
        """Build a frame with all P_Skip slices (for 'unchanged')."""
        if not self.sps:
            return []

        self.increment_frame_num()

        output_nals = []
        for row in range(self.height_mbs):
            first_mb = row * self.width_mbs
            output_nals.append(create_pskip_slice(
                first_mb, self.width_mbs, self.frame_num, self.sps, ref_ltr0=True
            ))

        return output_nals

    def build_spliced_frame(self, region_slices, region_info, target_sps, debug=False):
        """
        Build stitched frame with P_Skip + transplanted region slices.

        Args:
            region_slices: List of NAL units from region encoder
            region_info: (region_sps, region_pps, mb_x, mb_y, mb_w, mb_h)
            target_sps: SPS from full-frame encoder (for frame_num bits)
        """
        if not self.sps:
            return []

        region_sps, region_pps, region_mb_x, region_mb_y, region_mb_w, region_mb_h = region_info

        self.increment_frame_num()
        frame_num = self.frame_num

        # Group region slices by row
        region_rows = {}
        for nal in region_slices:
            reader = BitReader(nal.rbsp)
            first_mb = reader.read_ue()
            row = first_mb // region_mb_w
            region_rows[row] = nal

        if debug:
            print(f"[STITCH] region_slices={len(region_slices)}, region_rows={list(region_rows.keys())}")
            print(f"[STITCH] region: mb_pos=({region_mb_x},{region_mb_y}) mb_size=({region_mb_w},{region_mb_h})")

        output_nals = []
        transplanted_count = 0

        for row in range(self.height_mbs):
            first_mb = row * self.width_mbs

            # Check if this row intersects the region
            if region_mb_y <= row < region_mb_y + region_mb_h:
                region_local_row = row - region_mb_y

                # Left P_Skip (before region)
                if region_mb_x > 0:
                    output_nals.append(create_pskip_slice(
                        first_mb, region_mb_x, frame_num, self.sps, ref_ltr0=True
                    ))

                # Transplanted region slice
                if region_local_row in region_rows:
                    ov_nal = region_rows[region_local_row]
                    new_first_mb = first_mb + region_mb_x
                    is_idr = ov_nal.nal_unit_type == 5

                    rewritten = rewrite_slice_header(
                        ov_nal, region_sps, region_pps,
                        new_first_mb=new_first_mb,
                        new_idc=1,  # disable deblocking
                        new_frame_num=frame_num,
                        convert_idr_to_non_idr=is_idr,
                        new_nal_ref_idc=2,
                        target_sps=target_sps
                    )
                    output_nals.append(rewritten)
                    transplanted_count += 1

                    if debug and row == region_mb_y:  # First transplanted row
                        # Verify the rewritten slice
                        reader = BitReader(rewritten.rbsp)
                        out_first_mb = reader.read_ue()
                        out_slice_type = reader.read_ue()
                        print(f"[STITCH] First transplant: row={row} new_first_mb={new_first_mb} "
                              f"out_first_mb={out_first_mb} out_type={rewritten.nal_unit_type} "
                              f"slice_type={out_slice_type} orig_type={ov_nal.nal_unit_type}")
                else:
                    # No slice for this row, use P_Skip
                    output_nals.append(create_pskip_slice(
                        first_mb + region_mb_x, region_mb_w, frame_num, self.sps, ref_ltr0=True
                    ))
                    if debug:
                        print(f"[STITCH] WARNING: Missing region slice for local_row={region_local_row}")

                # Right P_Skip (after region)
                right_start = region_mb_x + region_mb_w
                if right_start < self.width_mbs:
                    output_nals.append(create_pskip_slice(
                        first_mb + right_start, self.width_mbs - right_start, frame_num, self.sps, ref_ltr0=True
                    ))
            else:
                # Full row P_Skip
                output_nals.append(create_pskip_slice(
                    first_mb, self.width_mbs, frame_num, self.sps, ref_ltr0=True
                ))

        if debug:
            print(f"[STITCH] output_nals={len(output_nals)}, transplanted={transplanted_count}")

        return output_nals


class LiveSpliceHandler(BaseFrameHandler):
    """
    POC 7: Live encoder using two-encoder architecture.

    Routes frames to appropriate encoder based on category:
    - "changed" -> Encoder Context 1 (full-frame)
    - "video_only" -> Encoder Context 2 (region) + stitched frame
    - "unchanged" -> Stitched P_Skip frame
    """

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height

        # Ensure even dimensions for x264
        self.enc_width = width if width % 2 == 0 else width + 1
        self.enc_height = height if height % 2 == 0 else height + 1

        self.width_mbs = (self.enc_width + 15) // 16
        self.height_mbs = (self.enc_height + 15) // 16

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.h264_path = self.output_dir / "live_spliced.h264"
        self.metrics_path = self.output_dir / "poc7_metrics.json"

        # Initialize encoder contexts
        self.encoder1 = FullFrameEncoder(self.enc_width, self.enc_height, self.output_dir)
        self.encoder2 = RegionEncoder(self.output_dir)
        self.frame_builder = StitchedFrameBuilder(self.width_mbs, self.height_mbs)

        # State
        self.frame_count = 0
        self.output_frame_count = 0
        self.start_time = None
        self.category_counts = Counter()
        self.output_type_counts = Counter()

        # Track when Encoder 2 needs IDR (after a "changed" frame breaks its reference chain)
        self.encoder2_needs_idr = True  # Start with IDR

        # Track last video_rect for detecting "changed" frames that are actually video-only
        self.last_video_rect = None

        # Output file
        self.output_file = None
        self.total_bytes = 0
        self.sps_pps_written = False

        # Map file for frame metadata
        self.map_path = self.output_dir / "poc7_framemap.csv"
        self.map_file = None
        self.current_phase = "init"
        self.drop_count = 0

        print(f"[poc7] Two-encoder architecture initialized")
        print(f"[poc7] Frame: {width}x{height}, Encoding: {self.enc_width}x{self.enc_height}")
        print(f"[poc7] MBs: {self.width_mbs}x{self.height_mbs}")
        print(f"[poc7] Chrome height: {chrome_height}")

    def _pad_pixels(self, pixels):
        """Pad pixels to encoding dimensions if needed."""
        if pixels.shape[1] != self.enc_width or pixels.shape[0] != self.enc_height:
            padded = np.zeros((self.enc_height, self.enc_width, 4), dtype=np.uint8)
            padded[:pixels.shape[0], :pixels.shape[1]] = pixels
            return padded
        return pixels

    def _is_damage_same_as_video(self, evt, tolerance=8):
        """Check if damage_rect matches last_video_rect (within tolerance)."""
        if not evt.damage_rects or not self.last_video_rect:
            return False
        if len(evt.damage_rects) != 1:
            return False  # Multiple damage rects = not just video

        dr = evt.damage_rects[0]
        vx, vy, vw, vh = self.last_video_rect

        # Adjust video_rect for chrome height (video_rect is CSS coords)
        vy_adjusted = vy + self.chrome_height

        # Check if damage_rect approximately matches video_rect
        if (abs(dr.x - vx) <= tolerance and
            abs(dr.y - vy_adjusted) <= tolerance and
            abs(dr.width - vw) <= tolerance and
            abs(dr.height - vh) <= tolerance):
            return True
        return False

    def _handle_video_only_from_damage(self, evt, pixels):
        """Handle a 'changed' frame that's actually video-only (damage matches video_rect)."""
        # Use last_video_rect since evt doesn't have video_rect for "changed" category
        sps, pps = self.encoder1.get_sps_pps()
        if sps is None:
            self._handle_changed(pixels)
            return

        vx, vy, vw, vh = self.last_video_rect
        vy = vy + self.chrome_height

        # Clamp to frame bounds
        vx = max(0, min(vx, self.enc_width - 1))
        vy = max(0, min(vy, self.enc_height - 1))
        vw = min(vw, self.enc_width - vx)
        vh = min(vh, self.enc_height - vy)

        if vw < 32 or vh < 32:
            self._handle_unchanged(pixels)
            return

        # Encode region with Encoder 2 (same as video_only)
        slices, region_info = self.encoder2.encode_region(pixels, vx, vy, vw, vh,
                                                          force_idr=self.encoder2_needs_idr)
        self.encoder2_needs_idr = False

        if slices and region_info:
            nals = self.frame_builder.build_spliced_frame(slices, region_info, sps)
            if nals:
                self._write_nals(nals)
                self.encoder1.notify_stitched_frame()
                self.output_type_counts['stitched_region'] += 1
                return

        self._handle_unchanged(pixels)

    def _write_map_entry(self, evt):
        """Write frame metadata to map file."""
        if self.map_file is None:
            self.map_file = open(self.map_path, "w")
            self.map_file.write("frame,category,rect_info,phase,drop_count,pixel_hash,enc_hash,out_hash\n")

        # Get rect info
        if evt.video_rect:
            rect_info = f"{evt.video_rect[0]}:{evt.video_rect[1]}:{evt.video_rect[2]}:{evt.video_rect[3]}"
        elif evt.damage_rects:
            rects = [f"{r.x}:{r.y}:{r.width}:{r.height}" for r in evt.damage_rects]
            rect_info = "|".join(rects)
        else:
            rect_info = "-"

        # Get pixel hash (stored by on_frame)
        pixel_hash = getattr(self, '_current_pixel_hash', '-')
        # Get encoder output hash (stored by handlers)
        enc_hash = getattr(self, '_current_enc_hash', '-')
        # Get final output hash (stored by _write_nals)
        out_hash = getattr(self, '_current_out_hash', '-')

        self.map_file.write(f"{self.output_frame_count},{evt.category},{rect_info},{self.current_phase},{self.drop_count},{pixel_hash},{enc_hash},{out_hash}\n")

    def on_phase_start(self, phase_name):
        """Called when a new phase starts."""
        print(f"[poc7] Phase: {phase_name}")
        self.current_phase = phase_name
        self.drop_count = 0

    def set_drop_count(self, count):
        """Set drop count for current phase."""
        self.drop_count = count

    def _write_nals(self, nals, include_sps_pps=False):
        """Write NAL units to output file."""
        import hashlib
        if self.output_file is None:
            self.output_file = open(self.h264_path, "wb")

        frame_bytes = b''
        for nal in nals:
            # Skip SPS/PPS from encoder output if we've already written them
            # (except when explicitly including them after IDR)
            if not include_sps_pps and nal.nal_unit_type in (7, 8) and self.sps_pps_written:
                continue

            # Modify SPS to allow 2 reference frames (needed for LTR 0 + short-term)
            if nal.nal_unit_type == 7:
                nal = modify_sps_num_ref_frames(nal, 2)

            data = b'\x00\x00\x00\x01' + nal.to_bytes()
            self.output_file.write(data)
            self.total_bytes += len(data)
            frame_bytes += data

            if nal.nal_unit_type in (7, 8):
                self.sps_pps_written = True

        # Hash of all bytes written for this frame
        self._current_out_hash = hashlib.md5(frame_bytes).hexdigest()[:8]

        self.output_frame_count += 1

        # Write map entry for this output frame
        if hasattr(self, '_current_evt') and self._current_evt:
            self._write_map_entry(self._current_evt)

    def on_frame(self, evt, pixels, info):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1
        self.category_counts[evt.category] += 1

        # Store event for map file (written when output is produced)
        self._current_evt = evt

        # Compute pixel hash for map file
        import hashlib
        if pixels is not None:
            self._current_pixel_hash = hashlib.md5(pixels.tobytes()).hexdigest()[:8]
        else:
            self._current_pixel_hash = '-'

        # Handle unchanged frames even without pixels (P_Skip doesn't need them)
        if evt.category == "unchanged":
            self._handle_unchanged(pixels)
            return

        # Other categories require pixels
        if pixels is None:
            return

        pixels = self._pad_pixels(pixels)

        # Route based on category
        if evt.category == "video_only":
            self.last_video_rect = evt.video_rect
            self._handle_video_only(evt, pixels)
        elif evt.category == "changed" and self._is_damage_same_as_video(evt):
            # "changed" but damage matches video_rect - treat as video_only
            self._handle_video_only_from_damage(evt, pixels)
        else:  # truly "changed"
            self._handle_changed(pixels)

        # Status every 100 frames
        if self.frame_count % 100 == 0:
            elapsed = time.monotonic() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[poc7] Frame {self.frame_count} ({fps:.1f} fps) - "
                  f"enc1: {self.encoder1.encode_count}, enc2: {self.encoder2.encode_count}, "
                  f"stitched: {self.output_type_counts['stitched_pskip'] + self.output_type_counts['stitched_region']}")

    def _handle_unchanged(self, pixels):
        """Handle 'unchanged' frame: emit full P_Skip stitched frame.

        P_Skip frames don't need pixel data - they just copy from reference.
        """
        # P_Skip frames have no encoder output (generated programmatically)
        self._current_enc_hash = 'pskip'

        # After an unchanged frame, encoder2's reference chain is broken
        # (decoder will use this P_Skip frame as reference, not encoder2's last output)
        self.encoder2_needs_idr = True

        sps, pps = self.encoder1.get_sps_pps()

        if sps is None:
            # No reference yet - need a changed frame first
            # Can't generate P_Skip without a reference
            if pixels is not None:
                self._handle_changed(self._pad_pixels(pixels))
            # else: skip this frame, we'll catch up when we get pixels
            return

        # Build stitched P_Skip frame
        nals = self.frame_builder.build_full_pskip_frame()
        if nals:
            self._write_nals(nals)
            self.encoder1.notify_stitched_frame()
            self.output_type_counts['stitched_pskip'] += 1

    def _handle_video_only(self, evt, pixels):
        """Handle 'video_only' frame: encode region + build stitched frame."""
        sps, pps = self.encoder1.get_sps_pps()

        if sps is None:
            # No reference yet, encode full frame first
            self._handle_changed(pixels)
            return

        if not evt.video_rect:
            # No video rect, treat as unchanged
            self._handle_unchanged(pixels)
            return

        # Get video region coordinates
        vx, vy, vw, vh = evt.video_rect

        # Adjust for chrome height (video_rect is in CSS viewport coords)
        vy = vy + self.chrome_height

        # Clamp to frame bounds
        vx = max(0, min(vx, self.enc_width - 1))
        vy = max(0, min(vy, self.enc_height - 1))
        vw = min(vw, self.enc_width - vx)
        vh = min(vh, self.enc_height - vy)

        if vw < 32 or vh < 32:
            # Region too small, treat as unchanged
            self._handle_unchanged(pixels)
            return

        # Debug: hash the video region pixels to detect duplicates
        import hashlib
        region_pixels = pixels[vy:vy+vh, vx:vx+vw]
        pixel_hash = hashlib.md5(region_pixels.tobytes()).hexdigest()[:8]

        # Encode region with Encoder Context 2
        # Force IDR if reference chain was broken by a "changed" frame
        slices, region_info = self.encoder2.encode_region(pixels, vx, vy, vw, vh,
                                                          force_idr=self.encoder2_needs_idr)
        self.encoder2_needs_idr = False  # Reset after encode

        # Compute encoder output hash for map file
        if slices:
            enc_bytes = b''.join(s.to_bytes() for s in slices)
            self._current_enc_hash = hashlib.md5(enc_bytes).hexdigest()[:8]
        else:
            self._current_enc_hash = '-'

        if slices and region_info:
            # Debug: check if slices are I or P
            i_slices = sum(1 for s in slices if s.nal_unit_type == 5)
            p_slices = sum(1 for s in slices if s.nal_unit_type == 1)
            if 400 <= self.frame_count <= 420:
                slice_types = [s.nal_unit_type for s in slices[:3]]
                print(f"[ENC2] frame={self.frame_count} slices: {len(slices)} total, {i_slices} IDR, {p_slices} non-IDR, first_types={slice_types}")

            # Build stitched frame with P_Skip + transplanted region
            debug_frame = (self.frame_count in (400, 401, 402, 540, 541, 542))
            nals = self.frame_builder.build_spliced_frame(slices, region_info, sps, debug=debug_frame)
            if nals:
                if debug_frame:
                    # Verify output NALs - hash the transplanted slices
                    transplant_bytes = b''
                    for n in nals:
                        reader = BitReader(n.rbsp)
                        first_mb = reader.read_ue()
                        slice_type = reader.read_ue()
                        if slice_type in (2, 7):  # I-slice (transplanted)
                            transplant_bytes += n.to_bytes()
                    transplant_hash = hashlib.md5(transplant_bytes).hexdigest()[:8]
                    print(f"[VERIFY] frame={self.frame_count} transplant_hash={transplant_hash} transplant_bytes={len(transplant_bytes)}")
                self._write_nals(nals)
                self.encoder1.notify_stitched_frame()
                self.output_type_counts['stitched_region'] += 1
                return

        # Fallback: treat as unchanged
        self._handle_unchanged(pixels)

    def _handle_changed(self, pixels):
        """Handle 'changed' frame: encode full frame with Encoder Context 1."""
        import hashlib
        # After a "changed" frame, Encoder 2's reference is stale - force IDR on next use
        self.encoder2_needs_idr = True

        # Get frame_num from frame_builder (increment first to get next frame_num)
        self.frame_builder.increment_frame_num()
        frame_num = self.frame_builder.frame_num

        nals = self.encoder1.encode(pixels, frame_num=frame_num)

        if nals:
            # Check if this is an IDR
            has_idr = any(n.nal_unit_type == 5 for n in nals)

            # Compute encoder output hash for map file
            enc_bytes = b''.join(n.to_bytes() for n in nals if n.nal_unit_type in (1, 5))
            self._current_enc_hash = hashlib.md5(enc_bytes).hexdigest()[:8]

            # Write with SPS/PPS for IDR frames
            self._write_nals(nals, include_sps_pps=has_idr)

            # Update frame builder state
            sps, pps = self.encoder1.get_sps_pps()
            self.frame_builder.set_sps(sps)

            if has_idr:
                self.frame_builder.reset_frame_num()
                self.output_type_counts['encoder1_idr'] += 1
            else:
                self.output_type_counts['encoder1_p'] += 1
        else:
            self._current_enc_hash = '-'

    # on_phase_start and set_drop_count defined earlier in class

    def close(self):
        if self.output_file:
            self.output_file.close()

        if self.map_file:
            self.map_file.close()

        # Close encoders
        self.encoder1.close()
        self.encoder2.close()

        elapsed = time.monotonic() - self.start_time if self.start_time else 0
        duration = self.output_frame_count / 25.0

        # Convert to MP4 with correct framerate (25fps)
        mp4_path = self.h264_path.with_suffix(".mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-r", "25",  # Force 25fps output
            "-i", str(self.h264_path),
            "-c:v", "copy",
            "-video_track_timescale", "25",
            str(mp4_path)
        ], capture_output=True)

        # Calculate metrics
        bitrate = (self.total_bytes * 8 / 1_000_000) / duration if duration > 0 else 0

        metrics = {
            "frame_count": self.frame_count,
            "output_frame_count": self.output_frame_count,
            "frame_width": self.width,
            "frame_height": self.height,
            "enc_width": self.enc_width,
            "enc_height": self.enc_height,
            "duration_sec": round(duration, 2),
            "total_bytes": self.total_bytes,
            "bitrate_mbps": round(bitrate, 3),
            "encoder1_encodes": self.encoder1.encode_count,
            "encoder1_idr": self.encoder1.idr_count,
            "encoder1_p": self.encoder1.p_count,
            "encoder1_ltr_refs": self.encoder1.ltr_ref_count,
            "encoder2_encodes": self.encoder2.encode_count,
            "category_counts": dict(self.category_counts),
            "output_type_counts": dict(self.output_type_counts),
            "encode_time_sec": round(elapsed, 2),
        }

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print()
        print("=" * 70)
        print("POC 7: Two-Encoder Architecture Summary (NVENC)")
        print("=" * 70)
        print(f"  Input frames:    {self.frame_count}")
        print(f"  Output frames:   {self.output_frame_count}")
        print(f"  Duration:        {duration:.2f} sec")
        print(f"  Output size:     {self.total_bytes:,} bytes")
        print(f"  Bitrate:         {bitrate:.3f} Mbps")
        print()
        print("  Frame Categories (input):")
        for cat, count in sorted(self.category_counts.items()):
            pct = 100 * count / self.frame_count if self.frame_count > 0 else 0
            print(f"    {cat:12s}: {count:6d} ({pct:5.1f}%)")
        print()
        print("  Output Types:")
        for op, count in sorted(self.output_type_counts.items()):
            pct = 100 * count / self.output_frame_count if self.output_frame_count > 0 else 0
            print(f"    {op:20s}: {count:6d} ({pct:5.1f}%)")
        print()
        print("  Encoder 1 (Full-Frame) Statistics:")
        print(f"    Total encodes:  {self.encoder1.encode_count}")
        print(f"    IDR frames:     {self.encoder1.idr_count}")
        print(f"    P frames:       {self.encoder1.p_count}")
        print(f"    LTR 0 refs:     {self.encoder1.ltr_ref_count}  (all P-frames reference LTR 0)")
        print()
        print("  Encoder 2 (Region) Statistics:")
        print(f"    Total encodes:  {self.encoder2.encode_count}")
        print(f"    IDR frames:     {self.encoder2.idr_count}")
        print(f"    P frames:       {self.encoder2.p_count}")
        print()
        stitched = self.output_type_counts.get('stitched_pskip', 0) + self.output_type_counts.get('stitched_region', 0)
        print(f"  Stitched frames:  {stitched}")
        print()
        print(f"  Output: {self.h264_path}")
        print(f"  MP4:    {mp4_path}")
        print(f"  Metrics: {self.metrics_path}")
        print()
        print("Playback:")
        print(f"  ffplay {mp4_path}")
        print(f"  python3 /home/wachtfogel/h264-newpoc/splicer/staging_player.py {mp4_path}")


def create_handler(fw, fh, chrome_height):
    return LiveSpliceHandler(fw, fh, chrome_height)
