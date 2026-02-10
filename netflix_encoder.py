#!/usr/bin/env python3
"""
Netflix Encoder: Layer-Side Classification + Dual Encoder (v4)

Both encoders run in the Vulkan layer. Python only does NAL post-processing:
1. Full-Frame Encoder (encoder1): Layer encodes on CHANGED frames.
   Python rewrites slices for LTR 0 marking.
2. Region Encoder (encoder2): Layer encodes on VIDEO_ONLY frames.
   Python extracts slices and builds stitched frames.

No pixel copy. No Python-side encoding. No prediction heuristic.

Usage:
    python3 ../chrome_gpu_tracer/test_frame_classifier.py \
        --handler netflix_encoder.py
"""

import json
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import av

# Add paths
TRACER_PATH = Path(__file__).parent.parent / "chrome_gpu_tracer"
SPLICER_PATH = Path("/home/wachtfogel/h264-newpoc/splicer")
sys.path.insert(0, str(TRACER_PATH))
sys.path.insert(0, str(SPLICER_PATH))

from frame_handler import BaseFrameHandler
from h264_splicer import (
    parse_annexb, NALUnit, BitWriter, BitReader, SPS, PPS,
    rewrite_slice_header, parse_slice_header,
    modify_sps_num_ref_frames,
)


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
    Encoder Context 1: Bitstream post-processor for "changed" frames.

    The Vulkan layer encodes the full frame. Python post-processes the
    bitstream to mark every frame as LTR 0 and reference previous LTR 0.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.width_mbs = (width + 15) // 16
        self.height_mbs = (height + 15) // 16

        # State tracking
        self.sps = None
        self.pps = None
        self.encode_count = 0
        self.idr_count = 0
        self.p_count = 0
        self.ltr_ref_count = 0

    def notify_stitched_frame(self):
        """Called when a stitched frame is output instead of encoder output."""
        pass

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
            marking = {'long_term': True}
        else:
            marking = {'mmco': [(6, 0)]}

        # Determine ref_pic_list_modification
        if ref_ltr0 and not is_idr:
            ref_mod = {'type': 'long_term', 'idx': 0}
        else:
            ref_mod = None

        rewritten = rewrite_slice_header(
            nal, self.sps, self.pps,
            new_first_mb=header.first_mb_in_slice,
            new_idc=header.disable_deblocking_filter_idc,
            new_frame_num=frame_num,
            new_ref_pic_list_mod=ref_mod,
            new_dec_ref_pic_marking=marking,
        )

        return rewritten

    def process_bitstream(self, bitstream, frame_num):
        """
        Post-process pre-encoded bitstream from layer.

        Takes pre-encoded bitstream, parses NALs, extracts SPS/PPS,
        rewrites slices with LTR 0 marking.

        Returns: list of NAL units
        """
        nals = parse_annexb(bitstream)

        output_nals = []
        for nal in nals:
            if nal.nal_unit_type == 7:
                self.sps = SPS.from_rbsp(nal.rbsp)
                output_nals.append(nal)
            elif nal.nal_unit_type == 8:
                self.pps = PPS.from_rbsp(nal.rbsp)
                output_nals.append(nal)
            elif nal.nal_unit_type in (1, 5):
                rewritten = self._rewrite_slice_for_ltr0(
                    nal, ref_ltr0=True, frame_num=frame_num)
                output_nals.append(rewritten)
            else:
                output_nals.append(nal)

        has_idr = any(n.nal_unit_type == 5 for n in output_nals)
        if has_idr:
            self.idr_count += 1
        else:
            self.p_count += 1
            self.ltr_ref_count += 1

        self.encode_count += 1
        return output_nals

    def get_sps_pps(self):
        """Return current SPS/PPS for stitched frame generation."""
        return self.sps, self.pps

    def close(self):
        pass


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

    def build_spliced_frame(self, region_slices, region_info, target_sps):
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

        output_nals = []

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
                else:
                    # No slice for this row, use P_Skip
                    output_nals.append(create_pskip_slice(
                        first_mb + region_mb_x, region_mb_w, frame_num, self.sps, ref_ltr0=True
                    ))

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

        return output_nals


class LiveSpliceHandler(BaseFrameHandler):
    """
    Netflix Encoder v4: Layer-side classification + dual encoder.

    The Vulkan layer classifies frames and encodes with the appropriate
    encoder. Python reads bitstreams from SHM and does NAL post-processing.

    Routes by frame_category from SHM:
    - CHANGED    -> Read encoder1 bitstream, LTR0 rewrite -> MKV
    - VIDEO_ONLY -> Read encoder2 bitstream, splice into stitched frame -> MKV
    - UNCHANGED  -> Stitched P_Skip frame (or VFR skip)
    """

    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.chrome_height = chrome_height

        # Ensure even dimensions for NVENC
        self.enc_width = width if width % 2 == 0 else width + 1
        self.enc_height = height if height % 2 == 0 else height + 1

        self.width_mbs = (self.enc_width + 15) // 16
        self.height_mbs = (self.enc_height + 15) // 16

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.output_dir / "metrics.json"

        # Encoder1 is now a bitstream post-processor only
        self.encoder1 = FullFrameEncoder(self.enc_width, self.enc_height)
        self.frame_builder = StitchedFrameBuilder(self.width_mbs, self.height_mbs)

        # Encoder2 SPS/PPS cache (extracted from layer bitstreams)
        self.enc2_sps = None
        self.enc2_pps = None
        self.enc2_count = 0
        self.enc2_idr_count = 0
        self.enc2_p_count = 0

        # State
        self.frame_count = 0
        self.output_frame_count = 0
        self.start_time = None
        self.category_counts = Counter()
        self.output_type_counts = Counter()

        # VFR support
        self.in_video_overlay_mode = False
        self.input_frame_index = 0

        # Output: direct VFR MKV via PyAV
        self.mkv_path = self.output_dir / "live_spliced.mkv"
        self.output_container = None
        self.output_stream = None
        self.total_bytes = 0
        self.extradata_set = False

        # Track encoder2 needs IDR (for stitched frame reference chain)
        self.encoder2_needs_idr = True

        print(f"[enc] v4: Layer-side classification + dual encoder")
        print(f"[enc] Frame: {width}x{height}, Encoding: {self.enc_width}x{self.enc_height}")
        print(f"[enc] MBs: {self.width_mbs}x{self.height_mbs}")
        print(f"[enc] Chrome height: {chrome_height}")
        print(f"[enc] No Python-side encoding — layer handles both encoders")

    def _write_nals(self, nals):
        """Write NAL units directly to VFR MKV."""
        frame_bytes = b''
        sps_data = None
        pps_data = None

        for nal in nals:
            if nal.nal_unit_type == 7:
                nal = modify_sps_num_ref_frames(nal, 2)
                sps_data = nal.to_bytes()
            elif nal.nal_unit_type == 8:
                pps_data = nal.to_bytes()

            data = b'\x00\x00\x00\x01' + nal.to_bytes()
            frame_bytes += data
            self.total_bytes += len(data)

        if self.output_container is None and sps_data and pps_data:
            self.output_container = av.open(str(self.mkv_path), mode="w")
            self.output_stream = self.output_container.add_stream("h264")
            self.output_stream.width = self.enc_width
            self.output_stream.height = self.enc_height
            self.output_stream.time_base = Fraction(1, 1000)
            self.output_stream.codec_context.extradata = (
                b'\x00\x00\x00\x01' + sps_data +
                b'\x00\x00\x00\x01' + pps_data
            )
            self.extradata_set = True

        if not frame_bytes or self.output_container is None:
            return

        pkt = av.Packet(frame_bytes)
        pkt.stream = self.output_stream
        pkt.pts = self.input_frame_index * 40  # 40ms per frame at 25fps
        pkt.dts = pkt.pts
        pkt.time_base = self.output_stream.time_base

        is_idr = any(n.nal_unit_type == 5 for n in nals)
        if is_idr:
            pkt.is_keyframe = True

        self.output_container.mux(pkt)
        self.output_frame_count += 1

    def on_frame(self, evt, pixels, info):
        """Process a frame. pixels is always None in v4. Bitstreams come via info."""
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.frame_count += 1
        self.input_frame_index = self.frame_count - 1
        self.category_counts[evt.category] += 1

        if evt.category == "unchanged":
            self._handle_unchanged()
        elif evt.category == "changed":
            self._handle_changed(info)
        elif evt.category == "video_only":
            self._handle_video_only(info)

        # Status every 100 frames
        if self.frame_count % 100 == 0:
            elapsed = time.monotonic() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            stitched = (self.output_type_counts.get('stitched_pskip', 0) +
                        self.output_type_counts.get('stitched_region', 0))
            print(f"[enc] Frame {self.frame_count} ({fps:.1f} fps) - "
                  f"enc1: {self.encoder1.encode_count}, enc2: {self.enc2_count}, "
                  f"stitched: {stitched}")

    def _handle_unchanged(self):
        """Handle 'unchanged' frame: VFR skip or P_Skip."""
        if self.in_video_overlay_mode:
            self.output_type_counts['vfr_skipped'] += 1
            return

        self.encoder2_needs_idr = True

        sps, pps = self.encoder1.get_sps_pps()
        if sps is None:
            return

        nals = self.frame_builder.build_full_pskip_frame()
        if nals:
            self._write_nals(nals)
            self.encoder1.notify_stitched_frame()
            self.output_type_counts['stitched_pskip'] += 1

    def _handle_changed(self, info):
        """Handle 'changed' frame: post-process encoder1 bitstream from layer."""
        self.encoder2_needs_idr = True

        if self.in_video_overlay_mode:
            print(f"[enc] Exiting video overlay mode at frame {self.frame_count}")
            self.in_video_overlay_mode = False

        if info is None or info.enc1_bitstream is None:
            return

        # Get frame_num from frame_builder
        self.frame_builder.increment_frame_num()
        frame_num = self.frame_builder.frame_num

        nals = self.encoder1.process_bitstream(info.enc1_bitstream, frame_num)

        if nals:
            has_idr = any(n.nal_unit_type == 5 for n in nals)
            self._write_nals(nals)

            sps, pps = self.encoder1.get_sps_pps()
            self.frame_builder.set_sps(sps)

            if has_idr:
                self.frame_builder.reset_frame_num()
                self.output_type_counts['encoder1_idr'] += 1
            else:
                self.output_type_counts['encoder1_p'] += 1

    def _handle_video_only(self, info):
        """Handle 'video_only' frame: extract encoder2 slices and splice."""
        if not self.in_video_overlay_mode:
            print(f"[enc] Entering video overlay mode at frame {self.frame_count}")
            self.in_video_overlay_mode = True

        sps, pps = self.encoder1.get_sps_pps()
        if sps is None:
            # No reference yet — need a changed frame first
            # Try to use encoder1 bitstream if available
            if info is not None and info.enc1_bitstream is not None:
                self._handle_changed(info)
            return

        if info is None or info.enc2_bitstream is None:
            self._handle_unchanged()
            return

        # Parse encoder2 bitstream
        nals = parse_annexb(info.enc2_bitstream)

        # Extract SPS/PPS
        for nal in nals:
            if nal.nal_unit_type == 7:
                self.enc2_sps = SPS.from_rbsp(nal.rbsp)
            elif nal.nal_unit_type == 8:
                self.enc2_pps = PPS.from_rbsp(nal.rbsp)

        if self.enc2_sps is None or self.enc2_pps is None:
            self._handle_unchanged()
            return

        # Get slice NALs
        slices = [n for n in nals if n.nal_unit_type in (1, 5)]

        # Track encoder2 frame types
        has_idr = any(n.nal_unit_type == 5 for n in nals)
        if has_idr:
            self.enc2_idr_count += 1
        else:
            self.enc2_p_count += 1
        self.enc2_count += 1

        # Get region MB coords from SHM
        mb_x, mb_y, mb_w, mb_h = info.enc2_region

        region_info = (self.enc2_sps, self.enc2_pps, mb_x, mb_y, mb_w, mb_h)

        # Build stitched frame
        output_nals = self.frame_builder.build_spliced_frame(slices, region_info, sps)
        if output_nals:
            self._write_nals(output_nals)
            self.encoder1.notify_stitched_frame()
            self.output_type_counts['stitched_region'] += 1
        else:
            self._handle_unchanged()

    def close(self):
        if self.output_container:
            self.output_container.close()

        self.encoder1.close()
        self._print_summary()

    def _print_summary(self):
        """Print summary after close."""
        duration = self.frame_count / 25.0
        elapsed = time.monotonic() - self.start_time if self.start_time else 0

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
            "encoder2_encodes": self.enc2_count,
            "encoder2_idr": self.enc2_idr_count,
            "encoder2_p": self.enc2_p_count,
            "category_counts": dict(self.category_counts),
            "output_type_counts": dict(self.output_type_counts),
            "encode_time_sec": round(elapsed, 2),
        }

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print()
        print("=" * 70)
        print("Netflix Encoder v4: Layer-Side Dual Encoder Summary")
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
        print("  Encoder 1 (Full-Frame, layer-side):")
        print(f"    Total encodes:  {self.encoder1.encode_count}")
        print(f"    IDR frames:     {self.encoder1.idr_count}")
        print(f"    P frames:       {self.encoder1.p_count}")
        print(f"    LTR 0 refs:     {self.encoder1.ltr_ref_count}")
        print()
        print("  Encoder 2 (Region, layer-side):")
        print(f"    Total encodes:  {self.enc2_count}")
        print(f"    IDR frames:     {self.enc2_idr_count}")
        print(f"    P frames:       {self.enc2_p_count}")
        print()
        stitched = (self.output_type_counts.get('stitched_pskip', 0) +
                    self.output_type_counts.get('stitched_region', 0))
        vfr_skipped = self.output_type_counts.get('vfr_skipped', 0)
        print(f"  Stitched frames:  {stitched}")
        print(f"  VFR skipped:      {vfr_skipped} (unchanged during video overlay)")
        print(f"  VFR mode:         {'YES' if vfr_skipped > 0 else 'NO'}")
        print()
        print(f"  Output: {self.mkv_path}")
        print(f"  Metrics: {self.metrics_path}")
        print()
        print("Playback:")
        print(f"  ffplay {self.mkv_path}")
        print(f"  python3 /home/wachtfogel/h264-newpoc/splicer/staging_player.py {self.mkv_path}")


def create_handler(fw, fh, chrome_height):
    return LiveSpliceHandler(fw, fh, chrome_height)
