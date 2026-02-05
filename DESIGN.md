# Netflix UI Encoder Design

## Overview

This document describes an H.264 encoding strategy optimized for the Netflix web UI.
The design leverages frame classification from chrome_gpu_tracer and MB-level splice
techniques from h264-newpoc to minimize both encoder burden and output bandwidth.

## Key Insight

The Netflix UI has three distinct activity patterns:

| Pattern | Frequency | Example |
|---------|-----------|---------|
| **Static** | Most frames | Background, sidebar, footer |
| **Video-only** | During playback | Preview thumbnails, trailers |
| **UI change** | Occasional | Hover effects, menu open, scroll |

Encoding the full 1920x1080 frame for every change wastes resources. Instead, we
encode only the regions that changed, using P_Skip blocks for static areas.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Chrome GPU Tracer                           │
│  (Vulkan capture layer + frame classifier)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ FrameEvent stream
                           │ - category: unchanged|video_only|changed
                           │ - damage_rects: [(x,y,w,h), ...]
                           │ - video_rect: (x,y,w,h) or None
                           │ - pixels: BGRA numpy array
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Frame Router                                │
│  Routes frames to appropriate encoder based on category         │
└────────────┬─────────────────────────────────┬──────────────────┘
             │ "changed"                       │ "video_only"
             ▼                                 ▼
┌────────────────────────────┐    ┌────────────────────────────────┐
│   Encoder Context 1        │    │   Encoder Context 2            │
│   (Full-Frame Encoder)     │    │   (Overlay/Region Encoder)     │
│                            │    │                                │
│   - Baseline profile       │    │   - Baseline profile           │
│   - Normal I/P frames      │    │   - Per-row slices             │
│   - Stores output as LTR 0 │    │   - No deblocking              │
│   - On resume: refs LTR 0  │    │   - ref=1, bframes=0           │
└────────────┬───────────────┘    └───────────────┬────────────────┘
             │                                    │
             │ Full encoded frames               │ Region slices
             ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Frame Assembler                             │
│  - For "changed": output full frame from Encoder 1              │
│  - For "video_only": build stitched frame:                      │
│    - P_Skip slices for unchanged areas (ref prev frame)         │
│    - Transplanted region slices from Encoder 2                  │
│  - For "unchanged": build full P_Skip frame (ref prev frame)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Complete H.264 frames
                           ▼
                      Output Stream
```

## Two-Encoder Architecture

The system uses **two separate encoder contexts** to handle different frame types efficiently:

### Encoder Context 1: Full-Frame Encoder

**Purpose:** Encode complete frames when significant UI changes occur ("changed" category).

**Characteristics:**
- Standard H.264 Baseline encoder
- Outputs I-frames (IDR) and P-frames normally
- Each output frame is stored as **Long-Term Reference 0 (LTR 0)**

**LTR 0 Usage:**
When we insert programmatically generated stitched frames into the output stream, Encoder 1's
internal short-term reference becomes stale (it didn't encode those frames). When we return
to Encoder 1 after a sequence of stitched frames, the first P-frame must reference LTR 0
instead of the normal short-term reference.

```
Timeline Example:
  Frame 100: [changed]    → Encoder 1 outputs P-frame, stores as LTR 0
  Frame 101: [video_only] → Stitched frame (refs prev frame 100)
  Frame 102: [video_only] → Stitched frame (refs prev frame 101)
  Frame 103: [video_only] → Stitched frame (refs prev frame 102)
  Frame 104: [changed]    → Encoder 1 outputs P-frame, MUST ref LTR 0 (not frame 103!)
                            Then stores this frame as new LTR 0
```

**Why LTR 0?**
- Encoder 1's internal state thinks frame 100 was the last frame it encoded
- But the decoder's last decoded frame is 103
- Without LTR 0, Encoder 1 would generate a P-frame referencing its stale state
- LTR 0 provides a known-good reference point that both encoder and decoder agree on

### Encoder Context 2: Overlay/Region Encoder

**Purpose:** Encode only the video region for "video_only" frames.

**Characteristics:**
- H.264 Baseline profile
- **Per-row slices:** `slice-max-mbs=<region_width_mbs>`
- **No deblocking:** Prevents cross-slice filtering artifacts at splice boundaries
- **Single reference:** `ref=1`
- **No B-frames:** `bframes=0`
- Outputs IDR and P-frames for the region only

**Output Usage:**
The encoded region slices are transplanted into programmatically generated stitched frames.
The IDR slices are converted to non-IDR (nal_unit_type 5→1) during transplantation.

### Programmatically Generated Stitched Frames

**Purpose:** Composite P_Skip areas with transplanted region slices.

**For "unchanged" frames:**
- Full frame of P_Skip slices
- Reference: previous decoded frame (normal short-term ref)
- Minimal bandwidth (just slice headers + mb_skip_run)

**For "video_only" frames:**
- P_Skip slices for areas outside the video region
- Transplanted region slices from Encoder 2 (converted IDR→non-IDR)
- Reference: previous decoded frame for P_Skip areas
- Region content is self-contained (intra-coded, just repackaged as non-IDR)

**Key Point:** Stitched frames reference the previous frame (can be another stitched frame
or a frame from Encoder 1). This is simpler than forcing LTR 0 reference everywhere.
The LTR 0 mechanism is specifically for when Encoder 1 needs to resume after stitched frames.

### Frame Flow by Category

```
"unchanged":
    └─→ Generate full P_Skip frame (refs prev) → Output

"video_only":
    ├─→ Encoder 2: encode video region
    └─→ Build stitched frame:
        - P_Skip for non-video areas (refs prev)
        - Transplanted region slices (intra content)
        └─→ Output

"changed":
    └─→ Encoder 1: encode full frame
        - If returning after stitched frames: ref LTR 0
        - Store output as LTR 0
        └─→ Output
```

### Reference Frame Summary

| Frame Source | References | Stored As |
|--------------|------------|-----------|
| Encoder 1 (first or after stitched) | LTR 0 | LTR 0 |
| Encoder 1 (consecutive) | Previous (normal) | LTR 0 |
| Stitched frame (P_Skip + region) | Previous frame | Not LTR (normal ref) |
| Full P_Skip (unchanged) | Previous frame | Not LTR (normal ref) |

## Frame Types

### Type 1: Unchanged Frame

**Trigger:** `evt.category == "unchanged"`

**Action:** Emit nothing. The decoder continues displaying the last decoded frame.
This is the most bandwidth-efficient case—zero bytes transmitted.

**Constraint:** Must track time since last emitted frame to avoid decoder timeout.
If >N seconds pass without output, emit a P-frame with all P_Skip.

### Type 2: Video-Only Frame

**Trigger:** `evt.category == "video_only"`

**Action:** Encode only the video rectangle region.

```
Full frame (1920x1080):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   P_Skip rows (static UI above video)                   │
│                                                         │
├────────┬────────────────────────────┬───────────────────┤
│P_Skip  │                            │ P_Skip            │
│columns │     Video Region           │ columns           │
│        │  (encoded with BG border)  │                   │
│        │                            │                   │
├────────┴────────────────────────────┴───────────────────┤
│                                                         │
│   P_Skip rows (static UI below video)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The video region is encoded as a separate small H.264 stream, then its slices
are transplanted into the output frame at the correct MB position.

### Type 3: Changed Frame

**Trigger:** `evt.category == "changed"`

**Action:** Encode all damage rectangles, merged if overlapping.

Multiple small changes can occur simultaneously (e.g., hover highlight + video
still updating). The encoder:

1. Collects all `damage_rects` from the event
2. Merges overlapping/adjacent rectangles
3. Expands each to MB alignment + 1 MB border
4. Encodes each region independently
5. Assembles output frame with P_Skip for untouched areas

### Type 4: I-Frame (Keyframe)

**Trigger:** Manual request or periodic interval (e.g., every 5 seconds)

**Action:** Encode full frame as IDR. Required for:
- Stream start
- Seek recovery
- Periodic refresh for join-in-progress clients

## Encoding Constraints

The two encoder contexts have different constraints based on their roles.

### Encoder Context 1: Full-Frame Encoder

**Purpose:** Encode complete frames for "changed" events.

**Constraints:**
```bash
ffmpeg -c:v libx264 -profile:v baseline \
    -x264opts "ref=1:bframes=0:ltr=1"
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| profile | baseline | CAVLC, no B-frames, simpler structure |
| ref | 1 | Single reference simplifies LTR management |
| bframes | 0 | No B-frames (baseline requirement) |
| ltr | 1 | Enable Long-Term Reference with 1 slot |

**LTR Usage:**
- Store each output frame as LTR 0
- When resuming after stitched frames, reference LTR 0

### Encoder Context 2: Overlay/Region Encoder

**Purpose:** Encode video regions for transplantation into stitched frames.

**Constraints:**
```bash
ffmpeg -c:v libx264 -profile:v baseline \
    -x264opts "slice-max-mbs=<region_width_mbs>:no-deblock:ref=1:bframes=0:subme=2"
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| profile | baseline | CAVLC, simple structure |
| slice-max-mbs | region_width_mbs | Per-row slices for transplantation |
| no-deblock | enabled | Prevents artifacts at splice boundaries |
| ref | 1 | Single reference |
| bframes | 0 | No B-frames |
| subme | 2 | Faster motion estimation (optional) |

**Critical: Per-Row Slices**

Each MB row must be a separate slice so we can transplant individual rows:
```
Region (5x3 MBs):
  Slice 0: MBs 0-4   (row 0)
  Slice 1: MBs 5-9   (row 1)
  Slice 2: MBs 10-14 (row 2)
```

### MB Alignment with BG Border

Regions encoded by Context 2 must include a 1 MB background border:

```python
def align_region(x, y, w, h, frame_pixels):
    """Expand region to MB boundaries plus 1 MB border."""
    # Align to 16-pixel boundaries
    mb_x = (x // 16) * 16
    mb_y = (y // 16) * 16
    mb_w = ((x + w + 15) // 16) * 16 - mb_x
    mb_h = ((y + h + 15) // 16) * 16 - mb_y

    # Add 1 MB (16px) border on all sides
    border_x = max(0, mb_x - 16)
    border_y = max(0, mb_y - 16)
    border_w = min(frame_w, mb_x + mb_w + 16) - border_x
    border_h = min(frame_h, mb_y + mb_h + 16) - border_y

    # Extract composite region from frame
    return frame_pixels[border_y:border_y+border_h,
                        border_x:border_x+border_w]
```

**Why the border?**
The border ensures that edge MBs in the region see identical neighbor pixels at
encode time and decode time, preventing visual artifacts at the splice boundary.
The border MBs themselves are not transplanted—only the inner region slices are used.

### No Deblocking for Region Encoder

```
x264: no-deblock
NVENC: disable_deblock=True
```

Deblocking filter operates across macroblock boundaries. When we splice region
slices into a P_Skip frame, the decoder would apply deblocking between the
transplanted MBs and the P_Skip MBs, causing visible seams. Disabling deblocking
for the region encoder prevents this.

**Note:** Encoder Context 1 (full-frame) CAN use deblocking since its output
is not spliced.

### Slice Header Rewriting for Transplantation

When transplanting region slices, the slice header must be rewritten:

| Field | Original | Rewritten |
|-------|----------|-----------|
| first_mb_in_slice | 0 (first row of region) | target_row * frame_width_mbs + region_mb_x |
| nal_unit_type | 5 (IDR) | 1 (non-IDR) |
| nal_ref_idc | varies | 2 (reference frame) |
| frame_num | 0 | current frame_num in output stream |
| idr_pic_id | present | removed (non-IDR) |
| dec_ref_pic_marking | IDR style | non-IDR style |
| disable_deblocking_filter_idc | varies | 1 (disabled) |

See `STITCHING_KNOWLEDGE.md` for detailed bitstream manipulation.

## Frame Assembly

### P_Skip Slice Generation

For rows with no dirty content, emit a single slice containing only `mb_skip_run`:

```python
def make_skip_slice(first_mb, mb_count):
    """Generate a P_Skip slice covering mb_count macroblocks."""
    # NAL header
    nal = bytearray([0x00, 0x00, 0x00, 0x01, 0x41])  # non-IDR P slice

    # Slice header (simplified)
    bits = BitWriter()
    bits.write_ue(first_mb)      # first_mb_in_slice
    bits.write_ue(0)             # slice_type = P
    bits.write_ue(0)             # pic_parameter_set_id
    bits.write_u(0, 4)           # frame_num (mod 16)

    # Slice data: single mb_skip_run
    bits.write_ue(mb_count)      # skip entire row

    return nal + bits.to_bytes()
```

### Slice Transplantation

When inserting an encoded region slice:

1. Parse the region slice to find `first_mb_in_slice`
2. Calculate new position: `target_row * frame_width_mbs + target_col`
3. Rewrite `first_mb_in_slice` in the bitstream
4. Append to output frame

```python
def transplant_slice(slice_data, region_mb_x, region_mb_y,
                     frame_width_mbs, row_offset):
    """Rewrite slice for insertion at target position."""
    # Parse existing first_mb_in_slice
    old_first_mb = parse_first_mb(slice_data)
    old_row = old_first_mb // region_width_mbs

    # Calculate new position
    new_row = row_offset + old_row
    new_first_mb = new_row * frame_width_mbs + region_mb_x

    # Rewrite in bitstream
    return rewrite_first_mb(slice_data, new_first_mb)
```

### Complete Frame Structure

```
Output Frame (P-frame):
┌──────────────────────────────────────────────────────────┐
│ NAL: SPS (if needed)                                     │
│ NAL: PPS (if needed)                                     │
├──────────────────────────────────────────────────────────┤
│ Slice 0: P_Skip row 0 (120 MBs)                          │
│ Slice 1: P_Skip row 1 (120 MBs)                          │
│ ...                                                      │
│ Slice 14: P_Skip row 14, cols 0-18 (19 MBs)              │
│ Slice 15: Video region row 0 (transplanted, 22 MBs)      │
│ Slice 16: P_Skip row 14, cols 41-119 (79 MBs)            │
│ ...                                                      │
│ Slice N: P_Skip row 67 (120 MBs)                         │
└──────────────────────────────────────────────────────────┘
```

## Bandwidth Analysis

### Baseline: Full-Frame Encoding

- Resolution: 1920x1080 @ 30fps
- Typical bitrate: 4-8 Mbps

### Optimized: Region-Only Encoding

**Video-only frames (most common during playback):**
- Video region: ~640x360 (common preview size)
- Region area: 22% of full frame
- Estimated bitrate: 1-2 Mbps + ~1KB P_Skip overhead

**UI change frames:**
- Typical damage: <5% of frame
- Hover effect: ~200x100 = 1% of frame
- Estimated bitrate: 50-200 Kbps per change

**Unchanged frames:**
- Zero bytes (or periodic P_Skip refresh)

**Estimated savings: 50-80% bandwidth reduction** depending on video activity.

## Reference Frame Management

### Overview

The two-encoder architecture requires careful reference frame management to ensure
the decoder always has a valid reference. The key mechanism is **Long-Term Reference 0 (LTR 0)**.

### LTR 0: The Bridge Between Encoder Contexts

**Problem:** When Encoder 1 encodes a "changed" frame, then we insert stitched frames,
Encoder 1's internal reference becomes stale. Its short-term reference buffer thinks
the last frame was the one it encoded, but the decoder's actual last frame is a stitched frame.

**Solution:** Encoder 1 always stores its output as LTR 0. When resuming after stitched
frames, it explicitly references LTR 0 instead of its stale short-term reference.

### Reference Chain Examples

**Consecutive "changed" frames (no stitching):**
```
Frame 100: [changed] → Encoder 1: I-frame, store as LTR 0
Frame 101: [changed] → Encoder 1: P-frame refs prev (100), store as LTR 0
Frame 102: [changed] → Encoder 1: P-frame refs prev (101), store as LTR 0
```
Normal operation, Encoder 1's short-term ref is valid.

**Mixed frames with stitching:**
```
Frame 100: [changed]    → Encoder 1: I-frame, store as LTR 0
Frame 101: [video_only] → Stitched: P_Skip + region slices, refs 100
Frame 102: [video_only] → Stitched: P_Skip + region slices, refs 101
Frame 103: [unchanged]  → Stitched: full P_Skip, refs 102
Frame 104: [changed]    → Encoder 1: P-frame, MUST ref LTR 0 (not its stale ref!)
                          Store as new LTR 0
Frame 105: [video_only] → Stitched: P_Skip + region slices, refs 104
```

**Key insight:** Stitched frames always reference the previous decoded frame (simple).
Only Encoder 1 needs the LTR 0 mechanism when resuming after stitched frames.

### Implementation: Bitstream Post-Processing for LTR 0

The encoder (NVENC or x264) doesn't need special LTR support. Instead, we **post-process
the encoded bitstream** to implement LTR 0 marking and referencing. This is the same
approach used for slice transplantation - rewriting slice headers after encoding.

**Two operations are needed:**

1. **Mark frame as LTR 0**: Modify `dec_ref_pic_marking()` in the slice header
2. **Reference LTR 0**: Modify `ref_pic_list_modification()` in the slice header

#### Marking a Frame as LTR 0

In `dec_ref_pic_marking()`, we add Memory Management Control Operation (MMCO) commands:

```
dec_ref_pic_marking() {
    if (nal_unit_type == 5) {  // IDR
        no_output_of_prior_pics_flag    u(1)
        long_term_reference_flag        u(1)  // Set to 1 for LTR
    } else {
        adaptive_ref_pic_marking_mode_flag  u(1)  // Set to 1
        // MMCO commands:
        memory_management_control_operation ue(v)  // 6 = mark current as LTR
        long_term_frame_idx                 ue(v)  // 0 = LTR index 0
        memory_management_control_operation ue(v)  // 0 = end
    }
}
```

**MMCO 6** = "Mark current picture as long-term reference with index N"

#### Referencing LTR 0 Instead of Short-Term Reference

When resuming after stitched frames, modify `ref_pic_list_modification()`:

```
ref_pic_list_modification() {
    ref_pic_list_modification_flag_l0   u(1)  // Set to 1
    // Modification commands:
    modification_of_pic_nums_idc        ue(v)  // 2 = use long_term_pic_num
    long_term_pic_num                   ue(v)  // 0 = LTR index 0
    modification_of_pic_nums_idc        ue(v)  // 3 = end
}
```

**modification_of_pic_nums_idc=2** = "Use long-term reference with index N"

#### Post-Processing Flow

```python
class FullFrameEncoder:
    def encode(self, pixels):
        # 1. Encode with NVENC (produces normal P-frame)
        bitstream = self._nvenc.encode(pixels)
        nals = parse_annexb(bitstream)

        # 2. Find slice NALs and rewrite headers
        for nal in nals:
            if nal.nal_unit_type in (1, 5):  # Slice
                if self.stitched_frames_since_last_encode > 0:
                    # Rewrite to reference LTR 0 instead of short-term ref
                    nal = rewrite_to_ref_ltr0(nal)

                # Rewrite to mark this frame as LTR 0
                nal = rewrite_to_mark_ltr0(nal)

        self.stitched_frames_since_last_encode = 0
        return nals
```

#### Why Bitstream Modification?

1. **Encoder independence**: Works with any encoder (NVENC, x264, etc.)
2. **Existing infrastructure**: Uses same slice header rewriting as transplantation
3. **No encoder API complexity**: Don't need per-frame encoder control
4. **Proven approach**: Same technique used successfully for splice operations

### Stitched Frame References

Stitched frames (for "unchanged" or "video_only") simply reference the previous
decoded frame using normal short-term reference. This works because:

1. P_Skip macroblocks copy from the co-located position in the reference
2. Transplanted region slices contain intra-coded content (IDR converted to non-IDR)
3. The reference chain is maintained: each stitched frame refs the previous frame

```python
def build_stitched_frame(region_slices, frame_num, sps):
    """Build frame with P_Skip + transplanted region slices."""
    # All slices reference previous frame (frame_num encodes this)
    # P_Skip: copies from prev
    # Region slices: intra content, no actual reference needed
    ...
```

### Scene Changes

When damage covers >30% of the frame, use Encoder 1 to encode a full frame
instead of attempting region splicing. This threshold balances:
- Quality: Large changes benefit from full-frame encoding
- Efficiency: Many overlapping regions have diminishing returns
- Complexity: Simpler to encode one full frame than many regions

## Integration with chrome_gpu_tracer

```python
from frame_handler import BaseFrameHandler
from region_encoder import RegionEncoder
from frame_assembler import FrameAssembler

class NetflixEncoderHandler(BaseFrameHandler):
    def __init__(self, width, height, chrome_height):
        self.width = width
        self.height = height
        self.encoder = RegionEncoder(width, height)
        self.assembler = FrameAssembler(width, height)
        self.frame_count = 0
        self.last_emit = 0

    def on_frame(self, evt, pixels, info):
        self.frame_count += 1

        if evt.category == "unchanged":
            # Emit nothing, but check for timeout refresh
            if self.frame_count - self.last_emit > 150:  # 5 sec @ 30fps
                self._emit_skip_frame()
            return

        if evt.category == "video_only":
            # Encode only video region
            regions = [self._expand_region(evt.video_rect)]
        else:
            # Encode all damage rects
            regions = self._merge_rects(evt.damage_rects)
            regions = [self._expand_region(r) for r in regions]

        # Check for scene change
        if self._total_area(regions) > 0.5 * self.width * self.height:
            self._emit_idr_frame(pixels)
            return

        # Encode regions and assemble frame
        encoded_regions = []
        for region in regions:
            crop = pixels[region.y:region.y2, region.x:region.x2]
            encoded = self.encoder.encode_region(crop, region)
            encoded_regions.append(encoded)

        frame_data = self.assembler.assemble(encoded_regions)
        self._emit_frame(frame_data)

    def _expand_region(self, rect):
        """Expand to MB alignment + 1 MB border."""
        # Implementation as shown above
        pass
```

## Open Questions

1. **Encoder latency:** NVENC has lower latency than x264 but requires GPU.
   Should we support both backends with automatic selection?

2. **Region encoder caching:** For stable video regions (same position across
   frames), can we reuse encoder state to improve compression?

3. **Multi-region frames:** When multiple small changes occur, is it better to
   merge into one region or encode separately? Tradeoff between encoder
   instances vs. P_Skip overhead.

4. **Audio sync:** This design focuses on video. How do we maintain A/V sync
   when emitting frames at irregular intervals?

## Testing & Playback

Use the Python player from `h264-newpoc` for testing output videos:

```bash
python3 ../h264-newpoc/splicer/staging_player.py <output_video>
```

**Player controls:**
| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right | Previous/Next frame |
| M | Toggle display mode (visible/staging/all) |
| R | Report artifact (after drawing bounding box) |
| Q | Quit |

**Why use this player instead of VLC/ffplay:**
- Frame-by-frame stepping to inspect splice boundaries
- Artifact reporting with bounding box selection (saves to `artifact_reports.txt`)
- Display mode toggle for staging frame workflows
- Precise frame number display

**Example testing workflow:**
```bash
# After generating POC output
python3 ../h264-newpoc/splicer/staging_player.py output/spliced_single.h264

# Step through frames with arrow keys
# If artifact found: draw box, press R, add notes
# Reports saved to artifact_reports.txt for debugging
```

---

## Implementation Phases

Each phase produces a working POC that can be tested independently. Later phases
build on earlier ones. Metrics are collected at each phase to measure improvement.

---

### POC 0: Baseline Metrics

**Goal:** Establish baseline encoding cost for full-frame encoding.

**Input:** Raw frame sequence from chrome_gpu_tracer (all frames, no classification)

**Output:** `baseline.mp4` + metrics log

**Implementation:**
```
poc0_baseline.py
├── Connect to chrome_gpu_tracer SHM
├── Capture N frames (e.g., 900 frames = 30 sec @ 30fps)
├── Encode with FFmpeg/x264 (VBR, CRF 23)
│   ffmpeg -f rawvideo -pix_fmt bgra -s 1920x1080 -r 30 \
│          -i pipe: -c:v libx264 -crf 23 baseline.mp4
└── Log: frame count, total bytes, encode time, avg bitrate
```

**Success criteria:**
- Produces playable MP4
- Logs baseline bitrate (expect 4-8 Mbps for Netflix UI)

**Files:**
- `poc0_baseline.py` — capture and encode script
- `output/baseline.mp4` — encoded video
- `output/baseline_metrics.json` — frame count, bitrate, timing

---

### POC 1: Frame Classification Logging

**Goal:** Verify chrome_gpu_tracer integration and understand frame distribution.

**Input:** Live chrome_gpu_tracer FrameEvent stream

**Output:** Classification log + statistics

**Implementation:**
```
poc1_classifier.py
├── Implement handler for chrome_gpu_tracer
├── For each frame, log:
│   ├── timestamp
│   ├── category (unchanged/video_only/changed)
│   ├── video_rect (if present)
│   └── damage_rects (list of rects)
└── Summarize: % unchanged, % video_only, % changed
```

**Success criteria:**
- During video preview hover: majority of frames are `video_only`
- During idle: majority are `unchanged`
- During scroll/navigation: bursts of `changed`

**Files:**
- `poc1_classifier.py` — handler that logs classifications
- `output/classification_log.jsonl` — per-frame classification data
- `output/classification_summary.txt` — statistics

---

### POC 2: Skip-Unchanged Encoding (VBR, No Stitching)

**Goal:** Reduce encoder load by skipping unchanged frames entirely.

**Input:** FrameEvent stream with pixels

**Output:** `skip_unchanged.mp4` with variable frame timing

**Implementation:**
```
poc2_skip_unchanged.py
├── Receive FrameEvent stream
├── If category == "unchanged":
│   └── Skip (don't encode)
├── Else:
│   ├── Write full frame to encoder
│   └── Track PTS for correct timing
└── Encode with FFmpeg VBR
    ffmpeg -f rawvideo -pix_fmt bgra -s 1920x1080 \
           -use_wallclock_as_timestamps 1 \
           -i pipe: -c:v libx264 -crf 23 -vsync vfr skip_unchanged.mp4
```

**Key insight:** VFR (variable frame rate) output. Decoder holds last frame until
next frame arrives, so skipping unchanged frames is visually correct.

**Success criteria:**
- Output file size < baseline (expect 30-50% smaller during idle periods)
- Video plays correctly with no visual glitches
- Encoder CPU usage lower during idle

**Metrics to compare vs POC 0:**
- Total bytes
- Frames encoded (should be << total frames)
- Encoder CPU time

**Files:**
- `poc2_skip_unchanged.py` — skip-unchanged encoder
- `output/skip_unchanged.mp4` — VFR encoded video
- `output/skip_unchanged_metrics.json` — comparison metrics

---

### POC 3: Region Extraction Analysis

**Goal:** Understand dirty region sizes and positions for optimization decisions.

**Input:** FrameEvent stream with damage_rects and video_rect

**Output:** Region statistics + sample region images

**Implementation:**
```
poc3_region_analysis.py
├── For each non-unchanged frame:
│   ├── Extract video_rect or damage_rects
│   ├── Calculate: area, % of frame, position
│   ├── Save sample crops as PNGs (first 100)
│   └── Log region metadata
└── Analyze:
    ├── Average region size
    ├── Region position distribution (heatmap)
    └── Potential savings from region-only encoding
```

**Success criteria:**
- Video regions typically <30% of frame area
- UI change regions typically <10% of frame area
- Clear visualization of where changes occur

**Files:**
- `poc3_region_analysis.py` — region extraction and analysis
- `output/regions/` — sample region crops
- `output/region_heatmap.png` — visualization of change locations
- `output/region_stats.json` — size/position statistics

---

### POC 4: Region-Only Encoding (Pixel Composite)

**Goal:** Encode only dirty regions, composite in pixel space.

**Input:** FrameEvent stream

**Output:** `region_composite.mp4`

**Method:** This is an intermediate step before bitstream splicing. We encode
only the dirty region, decode it, composite onto the previous frame in pixel
space, then re-encode the full frame. This validates region encoding without
bitstream manipulation.

**Implementation:**
```
poc4_region_composite.py
├── Maintain "last_frame" buffer (1920x1080)
├── For unchanged frames:
│   └── Output last_frame (or skip with VFR)
├── For video_only/changed frames:
│   ├── Extract dirty region + MB alignment + border
│   ├── Encode region as mini H.264:
│   │   ffmpeg -f rawvideo -s {region_w}x{region_h} ...
│   ├── Decode region back to pixels
│   ├── Composite onto last_frame at correct position
│   ├── Output composited frame
│   └── Update last_frame
└── Final encode to MP4
```

**Why this step?** Validates that:
1. Region extraction is correct
2. MB alignment preserves visual quality
3. Border handling works

**Success criteria:**
- Output visually identical to POC 2
- Region encoder handles various sizes correctly

**Files:**
- `poc4_region_composite.py` — region encode + pixel composite
- `output/region_composite.mp4` — output video
- `output/region_composite_metrics.json` — region sizes encoded

---

### POC 5: P_Skip Frame Generation

**Goal:** Generate valid H.264 P-frames containing only P_Skip macroblocks.

**Input:** SPS/PPS from a reference encode, frame dimensions

**Output:** Raw H.264 NAL units for all-skip P-frames

**Implementation:**
```
poc5_pskip_generator.py
├── BitWriter class for Exp-Golomb encoding
├── parse_sps_pps(reference.h264) → extract parameters
├── generate_skip_slice(first_mb, mb_count, frame_num):
│   ├── NAL header (nal_ref_idc=2, nal_unit_type=1)
│   ├── Slice header:
│   │   ├── first_mb_in_slice (ue)
│   │   ├── slice_type = 0 (P) (ue)
│   │   ├── pic_parameter_set_id (ue)
│   │   └── frame_num (u, log2_max_frame_num bits)
│   └── Slice data:
│       └── mb_skip_run = mb_count (ue)
├── generate_full_skip_frame(width_mbs, height_mbs, frame_num):
│   └── One slice per row, each with mb_skip_run = width_mbs
└── Test: append skip frames to I-frame, verify playback
```

**Test sequence:**
1. Encode single I-frame of static image
2. Generate 30 P_Skip frames
3. Concatenate: I + P + P + P + ...
4. Play in VLC/ffplay — should show static image for 1 second

**Success criteria:**
- Generated NAL units parse correctly (test with `h264_analyze`)
- Decoder accepts and displays frames
- No visual artifacts

**Files:**
- `poc5_pskip_generator.py` — P_Skip slice/frame generator
- `bitstream.py` — Exp-Golomb BitWriter/BitReader utilities
- `output/skip_test.h264` — test sequence
- `test_poc5.py` — unit tests for bitstream generation

---

### POC 6: Single-Region Splice (Overlay Injection)

**Goal:** Encode a region separately and splice its slices into a P_Skip frame.

**Input:** Background I-frame + changing region pixels

**Output:** Spliced H.264 stream with region overlaid on static background

**Implementation:**
```
poc6_single_region_splice.py
├── Encode background as I-frame (full 1920x1080)
├── For each frame with region change:
│   ├── Extract region with MB alignment + 1 MB border
│   ├── Encode region with constraints:
│   │   ├── Baseline profile
│   │   ├── slice-max-mbs = region_width_mbs (per-row slices)
│   │   ├── no-deblock
│   │   └── ref=1
│   ├── Parse region slices
│   ├── Build output frame:
│   │   ├── P_Skip slices for rows above region
│   │   ├── For each row intersecting region:
│   │   │   ├── P_Skip slice: cols 0 to region_start
│   │   │   ├── Transplanted region slice (rewrite first_mb_in_slice)
│   │   │   └── P_Skip slice: cols region_end to frame_width
│   │   └── P_Skip slices for rows below region
│   └── Emit assembled frame
└── Output: spliced.h264
```

**Region encoding command:**
```bash
ffmpeg -f rawvideo -pix_fmt bgra -s {w}x{h} -i region.raw \
    -c:v libx264 -profile:v baseline \
    -x264opts slice-max-mbs={w//16}:no-deblock:ref=1:bframes=0 \
    -f h264 region.h264
```

**Slice transplantation:**
```python
def transplant_slice(slice_nal, region_mb_x, region_mb_y,
                     frame_width_mbs, region_width_mbs):
    # Parse slice header to get row within region
    old_first_mb = parse_first_mb_in_slice(slice_nal)
    region_row = old_first_mb // region_width_mbs

    # Calculate position in full frame
    frame_row = region_mb_y + region_row
    new_first_mb = frame_row * frame_width_mbs + region_mb_x

    # Rewrite first_mb_in_slice
    return rewrite_first_mb_in_slice(slice_nal, new_first_mb)
```

**Success criteria:**
- Output plays correctly in decoder
- Region content appears at correct position
- No artifacts at region boundaries
- Bitrate << full-frame encoding

**Files:**
- `poc6_single_splice.py` — single region splice implementation
- `slice_parser.py` — parse H.264 slice headers
- `slice_transplant.py` — rewrite first_mb_in_slice
- `output/spliced_single.h264` — test output
- `test_poc6.py` — visual verification tests

---

### POC 7: Live Integration with chrome_gpu_tracer

**Goal:** Full integration using the two-encoder architecture with LTR 0 management.

**Input:** Live FrameEvent stream from Netflix browsing session

**Output:** Optimized H.264 stream using two encoder contexts

**Two-Encoder Architecture:**

```
poc7_live_splice_handler.py
├── Encoder Context 1 (Full-Frame Encoder):
│   ├── Standard x264 baseline encoder
│   ├── Used for "changed" frames only
│   ├── Stores each output as LTR 0
│   └── On resume after stitched frames: refs LTR 0
│
├── Encoder Context 2 (Region/Overlay Encoder):
│   ├── x264 with special constraints:
│   │   ├── slice-max-mbs=<region_width_mbs> (per-row slices)
│   │   ├── no-deblock
│   │   ├── ref=1, bframes=0
│   ├── Used for "video_only" frames only
│   └── Outputs region slices for transplantation
│
└── Frame Assembly:
    ├── "unchanged" → Full P_Skip frame (refs prev)
    ├── "video_only" → P_Skip + transplanted region (refs prev)
    └── "changed" → Full frame from Encoder 1 (refs LTR 0 if resuming)
```

**Frame Handling Logic:**

```python
def on_frame(evt, pixels):
    if evt.category == "unchanged":
        # Build full P_Skip frame referencing previous
        output_stitched_pskip_frame()
        encoder1.notify_stitched_frame()

    elif evt.category == "video_only":
        # Encoder 2: encode video region with special constraints
        region_slices = encoder2.encode_region(pixels, evt.video_rect)
        # Build stitched frame: P_Skip + transplanted region
        output_stitched_frame(region_slices)
        encoder1.notify_stitched_frame()

    elif evt.category == "changed":
        # Encoder 1: encode full frame
        # If returning after stitched frames, refs LTR 0
        full_frame = encoder1.encode(pixels)
        output_frame(full_frame)
        # Encoder 1 stores this as LTR 0 internally
```

**LTR 0 Management:**

```python
class FullFrameEncoder:
    def encode(self, pixels):
        if self.stitched_frames_since_last_output > 0:
            # CRITICAL: Reference LTR 0, not stale short-term ref
            return self._encode_with_ltr0_reference(pixels)
        else:
            return self._encode_normal(pixels)
```

**Run command:**
```bash
python3 chrome_gpu_tracer/test_frame_classifier.py \
    --handler poc7_live_splice_handler.py \
    --url https://www.netflix.com/browse
```

**Success criteria:**
- Smooth playback of captured session
- 50-80% bandwidth reduction vs baseline
- No visual artifacts during UI interactions
- Correct LTR 0 reference when Encoder 1 resumes after stitched frames
- Handles all Netflix UI patterns (browse, hover, scroll, video preview)

**Files:**
- `poc7_live_splice_handler.py` — two-encoder implementation
- `output/live_spliced.h264` — captured session
- `output/poc7_metrics.json` — bandwidth comparison

---

### Phase 8: Optimizations

**Goal:** Production-ready encoder with advanced features.

**Features:**

**8a. Long-Term Reference (LTR) for pause/resume:**
```python
# Mark frame as LTR before entering region-only mode
encoder.mark_ltr(frame_num)

# After exiting region-only mode, reference LTR for clean resume
encoder.set_ltr_reference(ltr_frame_num)
```

**8b. Scene change detection:**
```python
def is_scene_change(damage_rects, frame_area):
    damage_area = sum(r.w * r.h for r in damage_rects)
    return damage_area > 0.5 * frame_area  # >50% = scene change
```

**8c. Adaptive region encoding:**
- Small regions (<5% frame): encode as spliced region
- Medium regions (5-50%): encode as spliced region
- Large regions (>50%): encode as full I-frame

**8d. NVENC backend:**
```python
# Switch from x264 to NVENC for lower latency
encoder = NVENCRegionEncoder(
    slice_mode=2,  # MB row based
    disable_deblock=True,
    qp=23
)
```

**8e. Streaming output:**
- Fragment MP4 (fMP4) for live streaming
- WebSocket transport for browser playback
- RTMP output for OBS/streaming platforms

**Files:**
- `ltr_manager.py` — LTR frame management
- `scene_detector.py` — scene change detection
- `nvenc_region_encoder.py` — NVENC backend
- `streaming_output.py` — fMP4/WebSocket/RTMP output

---

## Summary: POC Progression

| POC | Input | Output | Key Learning |
|-----|-------|--------|--------------|
| 0 | Raw frames | Full encode | Baseline bitrate |
| 1 | FrameEvents | Log | Frame distribution |
| 2 | FrameEvents | VFR MP4 | Skip-unchanged savings |
| 3 | FrameEvents | Stats | Region size analysis |
| 4 | FrameEvents | Composite MP4 | Region encode validation |
| 5 | Dimensions | NAL units | P_Skip generation |
| 6 | BG + region | Spliced H.264 | Single splice |
| 7 | Live stream | Optimized H.264 | Full integration |
| 8 | Live stream | Production H.264 | Optimizations |

Each POC builds on the previous, allowing incremental testing and validation.
