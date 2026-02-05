# H.264 Slice Stitching Knowledge

This document captures the essential knowledge for transplanting H.264 slices from
a separately-encoded region into a composite frame. Based on `h264-newpoc/splicer/`.

## Core Concept

Instead of re-encoding the full frame when only a region changes, we:
1. Encode the changed region separately as H.264 with per-row slices
2. Generate P_Skip slices for unchanged areas (zero bytes, copy from previous frame)
3. Transplant the region slices into the output frame at the correct MB position

## Key Data Structures

### BitReader/BitWriter

Variable-length Exp-Golomb coding used throughout H.264:

```python
class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0  # 0-7, 0 is MSB

    def read_bits(self, n: int) -> int:
        """Read n bits MSB first."""
        result = 0
        for _ in range(n):
            byte = self.data[self.byte_pos]
            bit = (byte >> (7 - self.bit_pos)) & 1
            result = (result << 1) | bit
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.bit_pos = 0
                self.byte_pos += 1
        return result

    def read_ue(self) -> int:
        """Read unsigned Exp-Golomb coded value."""
        leading_zeros = 0
        while self.read_bits(1) == 0:
            leading_zeros += 1
        if leading_zeros == 0:
            return 0
        suffix = self.read_bits(leading_zeros)
        return (1 << leading_zeros) - 1 + suffix

    def read_se(self) -> int:
        """Read signed Exp-Golomb coded value."""
        ue = self.read_ue()
        if ue == 0:
            return 0
        sign = 1 if (ue & 1) else -1
        return sign * ((ue + 1) >> 1)


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current_byte = 0
        self.bit_pos = 0

    def write_bits(self, value: int, n: int):
        """Write n bits from value (MSB first)."""
        for i in range(n - 1, -1, -1):
            bit = (value >> i) & 1
            self.current_byte |= (bit << (7 - self.bit_pos))
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.data.append(self.current_byte)
                self.current_byte = 0
                self.bit_pos = 0

    def write_ue(self, value: int):
        """Write unsigned Exp-Golomb coded value."""
        if value == 0:
            self.write_bits(1, 1)
            return
        code = value + 1
        n = code.bit_length() - 1
        self.write_bits(0, n)
        self.write_bits(code, n + 1)

    def write_se(self, value: int):
        """Write signed Exp-Golomb coded value."""
        if value == 0:
            self.write_ue(0)
        elif value > 0:
            self.write_ue(2 * value - 1)
        else:
            self.write_ue(-2 * value)

    def write_remaining_bits(self, reader: BitReader):
        """Copy all remaining bits from reader."""
        if self.bit_pos == 0 and reader.bit_pos == 0:
            self.data.extend(reader.data[reader.byte_pos:])
        else:
            while reader.byte_pos < len(reader.data):
                bit = reader.read_bits(1)
                self.write_bits(bit, 1)

    def finish_rbsp(self) -> bytes:
        """Add rbsp_trailing_bits and return bytes."""
        self.write_bits(1, 1)  # rbsp_stop_one_bit
        while self.bit_pos != 0:
            self.write_bits(0, 1)
        return bytes(self.data)
```

### Emulation Prevention Bytes

H.264 NAL units must not contain `00 00 00`, `00 00 01`, `00 00 02`, or `00 00 03`
sequences (reserved for start codes). An emulation prevention byte `03` is inserted:

```python
def remove_emulation_prevention(data: bytes) -> bytes:
    """Remove 0x03 after 0x00 0x00."""
    result = bytearray()
    i = 0
    while i < len(data):
        if i + 2 < len(data) and data[i:i+3] == b'\x00\x00\x03':
            result.extend(b'\x00\x00')
            i += 3
        else:
            result.append(data[i])
            i += 1
    return bytes(result)


def add_emulation_prevention(rbsp: bytes) -> bytes:
    """Add 0x03 where needed."""
    result = bytearray()
    i = 0
    while i < len(rbsp):
        if i + 2 < len(rbsp) and rbsp[i:i+2] == b'\x00\x00' and rbsp[i+2] in (0, 1, 2, 3):
            result.extend(b'\x00\x00\x03')
            i += 2
        else:
            result.append(rbsp[i])
            i += 1
    return bytes(result)
```

### NAL Unit Structure

```python
@dataclass
class NALUnit:
    nal_ref_idc: int      # 2 bits (0=non-ref, 1-3=reference)
    nal_unit_type: int    # 5 bits (1=non-IDR slice, 5=IDR slice, 7=SPS, 8=PPS)
    rbsp: bytes           # Raw Byte Sequence Payload (emulation prevention removed)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'NALUnit':
        header = data[0]
        nal_ref_idc = (header >> 5) & 0x3
        nal_unit_type = header & 0x1f
        rbsp = remove_emulation_prevention(data[1:])
        return cls(nal_ref_idc, nal_unit_type, rbsp)

    def to_bytes(self) -> bytes:
        header = (self.nal_ref_idc << 5) | self.nal_unit_type
        rbsp_with_ep = add_emulation_prevention(self.rbsp)
        return bytes([header]) + rbsp_with_ep
```

## SPS Parameters Needed for Slice Rewriting

```python
@dataclass
class SPS:
    profile_idc: int
    level_idc: int
    seq_parameter_set_id: int
    log2_max_frame_num_minus4: int        # frame_num uses this many + 4 bits
    pic_order_cnt_type: int               # 0, 1, or 2
    log2_max_pic_order_cnt_lsb_minus4: int  # Only if poc_type == 0
    pic_width_in_mbs_minus1: int
    pic_height_in_map_units_minus1: int
    frame_mbs_only_flag: int
```

Critical: `log2_max_frame_num_minus4` determines how many bits `frame_num` occupies
in the slice header. Both source and target SPS must match, or you must adjust.

## Slice Header Structure (Baseline Profile)

Order of fields in slice header for P-slice (nal_unit_type=1):

```
1. first_mb_in_slice (ue)         ← MUST REWRITE for position
2. slice_type (ue)                ← Keep as-is (0=P, 2=I, 5=P-all, 7=I-all)
3. pic_parameter_set_id (ue)      ← Keep as-is (usually 0)
4. frame_num (u, N bits)          ← MUST REWRITE to match output frame
5. [if IDR] idr_pic_id (ue)       ← Only for nal_unit_type=5
6. [if poc_type=0] pic_order_cnt_lsb (u, M bits)
7. [if P/B] num_ref_idx_active_override_flag (1 bit)
8. [if override] num_ref_idx_l0_active_minus1 (ue)
9. [if P/B] ref_pic_list_modification section
10. [if nal_ref_idc>0] dec_ref_pic_marking section
11. slice_qp_delta (se)
12. [if deblock_control] disable_deblocking_filter_idc (ue)
13. [if idc != 1] slice_alpha_c0_offset_div2 (se)
14. [if idc != 1] slice_beta_offset_div2 (se)
15. SLICE DATA (macroblock layer)
```

## The Core Rewrite Function

The key insight: you must parse the ENTIRE slice header to know where each field is,
then rewrite specific fields while copying others bit-by-bit.

```python
def rewrite_slice_header(
    nal: NALUnit,
    sps: SPS,
    pps: PPS,
    new_first_mb: int,
    new_idc: int,
    new_frame_num: int = None,
    convert_idr_to_non_idr: bool = False,
    new_nal_ref_idc: int = None,
    target_sps: SPS = None
) -> NALUnit:
    """
    Rewrite slice header for transplantation.

    Key rewrites:
    - first_mb_in_slice: Position in output frame
    - frame_num: Must match output frame's frame_num
    - disable_deblocking_filter_idc: Usually set to 1 (disable)
    - nal_ref_idc: Set to 2 for reference frames
    - Convert IDR to non-IDR if needed (removes idr_pic_id, changes dec_ref_pic_marking)
    """
    # 1. Parse original header to get all field values and positions
    header, reader = parse_slice_header(nal, sps, pps)

    # 2. Determine output NAL type
    output_nal_type = nal.nal_unit_type
    if convert_idr_to_non_idr and nal.nal_unit_type == 5:
        output_nal_type = 1  # IDR → non-IDR

    # 3. Write new header
    writer = BitWriter()

    # Write rewritten fields
    writer.write_ue(new_first_mb)
    writer.write_ue(header.slice_type)
    writer.write_ue(header.pic_parameter_set_id)

    # frame_num with correct bit width
    output_sps = target_sps if target_sps else sps
    frame_num_bits = output_sps.log2_max_frame_num_minus4 + 4
    frame_num_to_write = new_frame_num if new_frame_num is not None else header.frame_num
    writer.write_bits(frame_num_to_write, frame_num_bits)

    # idr_pic_id only if still IDR
    if output_nal_type == 5:
        writer.write_ue(header.idr_pic_id)

    # Copy pic_order_cnt_lsb if present
    if sps.pic_order_cnt_type == 0:
        poc_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4
        writer.write_bits(header.pic_order_cnt_lsb, poc_bits)

    # Copy intermediate sections (num_ref_idx_override, ref_pic_list_mod)
    # by reading from original and writing bit-by-bit
    # ... (complex bit copying logic)

    # Write dec_ref_pic_marking
    if convert_idr_to_non_idr:
        writer.write_bits(0, 1)  # adaptive_ref_pic_marking_mode_flag = 0
    else:
        # Copy original dec_ref_pic_marking
        pass

    # Copy slice_qp_delta
    writer.write_se(header.slice_qp_delta)

    # Write new deblocking params
    writer.write_ue(new_idc)  # disable_deblocking_filter_idc
    if new_idc != 1:
        writer.write_se(header.slice_alpha_c0_offset_div2)
        writer.write_se(header.slice_beta_offset_div2)

    # Copy remaining slice data bit-by-bit
    writer.write_remaining_bits(reader)

    return NALUnit(
        nal_ref_idc=new_nal_ref_idc if new_nal_ref_idc else nal.nal_ref_idc,
        nal_unit_type=output_nal_type,
        rbsp=writer.finish()
    )
```

## P_Skip Slice Generation

For unchanged regions, generate slices with only `mb_skip_run`:

```python
def create_bg_skip_slice(row, first_mb_offset, width_mbs, frame_num, sps, nal_ref_idc=2):
    """Create P-slice with all P_Skip macroblocks."""
    w = BitWriter()

    first_mb = row * frame_width_mbs + first_mb_offset

    # Slice header
    w.write_ue(first_mb)           # first_mb_in_slice
    w.write_ue(0)                  # slice_type = P
    w.write_ue(0)                  # pic_parameter_set_id
    w.write_bits(frame_num, sps.log2_max_frame_num_minus4 + 4)

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
    w.write_ue(width_mbs)

    rbsp = w.finish_rbsp()
    return NALUnit(nal_ref_idc, 1, rbsp)  # nal_unit_type=1 (non-IDR)
```

## Frame Composition

For each output frame row:

```python
def build_output_frame(overlay_slices, overlay_sps, bg_sps, frame_num,
                       target_col_mb, target_row_mb, padded_width_mbs, padded_height_mbs):
    nals = []

    padded_col_mb = target_col_mb - PADDING_MBS
    padded_row_mb = target_row_mb - PADDING_MBS
    padded_end_row_mb = padded_row_mb + padded_height_mbs
    padded_end_col_mb = padded_col_mb + padded_width_mbs

    # Group overlay slices by row
    ov_rows = {parse_first_mb(s) // padded_width_mbs: s for s in overlay_slices}

    for row in range(frame_height_mbs):
        ov_local_row = row - padded_row_mb

        if padded_row_mb <= row < padded_end_row_mb and ov_local_row in ov_rows:
            # Row contains overlay

            # Left P_Skip
            if padded_col_mb > 0:
                nals.append(create_bg_skip_slice(row, 0, padded_col_mb, frame_num, bg_sps))

            # Transplanted overlay slice
            ov_nal = ov_rows[ov_local_row]
            new_first_mb = row * frame_width_mbs + padded_col_mb
            rewritten = rewrite_slice_header(
                ov_nal, overlay_sps, overlay_pps,
                new_first_mb=new_first_mb,
                new_idc=1,  # Disable deblocking
                new_frame_num=frame_num,
                convert_idr_to_non_idr=True,
                new_nal_ref_idc=2,
                target_sps=bg_sps
            )
            nals.append(rewritten)

            # Right P_Skip
            if padded_end_col_mb < frame_width_mbs:
                nals.append(create_bg_skip_slice(
                    row, padded_end_col_mb,
                    frame_width_mbs - padded_end_col_mb,
                    frame_num, bg_sps
                ))
        else:
            # Full BG row
            nals.append(create_bg_skip_slice(row, 0, frame_width_mbs, frame_num, bg_sps))

    return nals
```

## Encoding Constraints for Overlay

The overlay region MUST be encoded with these constraints:

### x264
```bash
ffmpeg -i region.raw \
    -c:v libx264 \
    -profile:v baseline \
    -x264opts slice-max-mbs=<width_mbs>:subme=2:no-deblock:ref=1:bframes=0 \
    region.h264
```

### NVENC
```python
encoder = NVENCEncoder(
    width, height,
    slice_mode=2,        # MB row based
    slice_mode_data=1,   # 1 MB row per slice
    disable_deblock=True,
    qp=23
)
```

Key settings:
- **slice-max-mbs / slice_mode=2** — One slice per MB row for transplantability
- **no-deblock / disable_deblock** — Prevents cross-slice filtering artifacts
- **ref=1** — Single reference frame
- **bframes=0** — No B-frames (Baseline profile)
- **subme=2** — Limits motion search (x264 specific)

## Critical Gotchas

### 1. frame_num Mismatch
The transplanted slice's `frame_num` MUST match the output frame's `frame_num`.
If the overlay was encoded starting from frame 0, but you're inserting into
output frame 5, you must rewrite frame_num to 5.

### 2. IDR to Non-IDR Conversion
IDR slices (nal_unit_type=5) force the decoder to flush references. When
transplanting IDR slices into a non-IDR frame:
- Change nal_unit_type from 5 to 1
- Remove idr_pic_id field
- Change dec_ref_pic_marking from IDR format (2 bits) to non-IDR format (1 bit)

### 3. Reference Frame Chain
All output frames must have `nal_ref_idc > 0` (typically 2) so P_Skip blocks
in the next frame can reference them. If you output a non-reference frame,
the next frame's P_Skip will fail.

### 4. BG Padding
Always encode the overlay region with 1 MB (16 pixels) of background padding
on all sides. This ensures edge MBs see the same neighbor pixels during
encoding and decoding.

### 5. pic_parameter_set_id
The transplanted slice must reference the same PPS as the rest of the frame.
Usually both are 0, but if different, you must rewrite this field too.

### 6. log2_max_frame_num Mismatch
If source and target SPS have different `log2_max_frame_num_minus4`, the
frame_num field has different bit widths. You must account for this when
rewriting.

## File References

- `h264-newpoc/splicer/h264_splicer.py` — Core BitReader/BitWriter, NAL parsing, rewrite_slice_header()
- `h264-newpoc/splicer/basic/hybrid_overlay_test.py` — Complete example of frame composition
- `h264-newpoc/LEARNINGS.md` — High-level constraints and workflow

## Usage Example

```python
from h264_splicer import (
    parse_annexb, NALUnit, BitWriter, BitReader, SPS, PPS,
    rewrite_slice_header,
)

# Parse background
bg_nals = parse_annexb(open('bg.h264', 'rb').read())
bg_sps = SPS.from_rbsp(next(n for n in bg_nals if n.nal_unit_type == 7).rbsp)

# Parse overlay (encoded with per-row slices)
ov_nals = parse_annexb(open('overlay.h264', 'rb').read())
ov_sps = SPS.from_rbsp(next(n for n in ov_nals if n.nal_unit_type == 7).rbsp)
ov_slices = [n for n in ov_nals if n.nal_unit_type in (1, 5)]

# Build output frame
output_nals = []
for row in range(frame_height_mbs):
    if row in overlay_rows:
        # Transplant overlay slice
        rewritten = rewrite_slice_header(
            ov_slices[row - overlay_start_row],
            ov_sps, ov_pps,
            new_first_mb=row * frame_width_mbs + overlay_col_mb,
            new_idc=1,
            new_frame_num=current_frame_num,
            convert_idr_to_non_idr=True,
            new_nal_ref_idc=2,
            target_sps=bg_sps
        )
        output_nals.append(rewritten)
    else:
        # P_Skip row
        output_nals.append(create_bg_skip_slice(row, 0, frame_width_mbs, current_frame_num, bg_sps))

# Write output
with open('output.h264', 'wb') as f:
    for nal in output_nals:
        f.write(b'\x00\x00\x00\x01')
        f.write(nal.to_bytes())
```
