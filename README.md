# Netflix Encoder

Live H.264 encoding for Chrome's Netflix UI with VFR (variable frame rate) output via two-encoder architecture.

## Versions

- **Chrome**: 145.0.7632.75
- **NVIDIA GPU**: GeForce RTX 3050
- **CUDA**: 11.x+ (dlopen'd at runtime)
- **NVENC**: libnvidia-encode.so.1
- **Vulkan**: 1.2+
- **Python**: 3.10+
- **FFmpeg**: 6.1.1

## Architecture

### Two-Encoder System

1. **Encoder1 (Full-Frame, GPU via Vulkan layer)**
   - Input: BGRA/ARGB from Chrome swapchain
   - Codec: H.264 Constrained Baseline, CAVLC
   - QP: 23 (constant)
   - Per-MB-row slices (for P-Skip frame generation)
   - Output: Full-frame IDR + P-frames with LTR 0 reference

2. **Encoder2 (Region, per-row slices)**
   - Input: Cropped video region (BGRA → NV12)
   - Codec: H.264 Constrained Baseline, CAVLC
   - QP: 23, per-row slices
   - Output: Region slices for transplantation

### Frame Classification

Three categories (via damage rects + video state from SHM):
- **UNCHANGED** (~20%): VFR skip, no encoding
- **VIDEO_ONLY** (~55%): P_Skip full frame + encoder2 region slices (stitched)
- **CHANGED** (~25%): Encoder1 full-frame output

### VFR Output

Skips unchanged frames during video overlay mode, extending previous frame duration in MKV. Reduces output bitrate while maintaining visual quality.

## Building

```bash
# Layer (C, Vulkan + CUDA interop)
cd vulkan_layer/build
cmake .. && make

# Python dependencies
pip install av numpy

# NVENC library (C++ wrapper)
cd /path/to/h264-newpoc/splicer
g++ -shared -fPIC -o nvenc_lib.so nvenc_lib.cpp -I/usr/include/ffnvcodec -ldl
```

## Running

```bash
# Full 10-phase Netflix UI test (47 seconds, 1132 frames)
python3 netflix_encoder.py

# Output
output/live_spliced.mkv (H.264, VFR, 4 MB)
output/metrics.json (frame stats)

# Playback
ffplay output/live_spliced.mkv
```

## Recent Improvements (Feb 2026)

- **Phase 1**: NVENC ARGB input support (eliminate BGRA→NV12 CPU conversion)
- **Phase 2a**: SHM v4 with encoder bitstream regions + video state fields
- **Phase 2b**: Layer-side encoder1 (GPU → NVENC in Vulkan layer)
- **Chrome 145**: Verified working with latest Chrome + Vulkan layer

## Future Enhancements

- **NVENC Vulkan Input**: Use `NV_ENC_INPUT_RESOURCE_TYPE_VULKAN_IMAGE` to hand VkImage directly to NVENC (no CUDA needed)
- **Zero-Copy**: Vulkan-CUDA interop for GPU-only pixel path (eliminate host staging)
- **Encoder2 in Layer**: Move region encoding to layer (requires video region signaling via SHM)
