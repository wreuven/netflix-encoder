/*
 * nvenc_encode.h -- NVENC encoder for the Vulkan capture layer.
 *
 * Encodes BGRA (ARGB in NVENC terms) frames on the GPU via CUDA + NVENC.
 * All GPU libraries are loaded via dlopen to avoid hard link dependencies.
 */

#ifndef NVENC_ENCODE_H
#define NVENC_ENCODE_H

#include <stddef.h>
#include <stdint.h>

typedef struct nvenc_ctx nvenc_ctx_t;

/*
 * Encoder configuration for nvenc_layer_init_config().
 */
typedef struct {
    int width, height, qp;
    int slice_mode;        /* 2=MB rows per slice, 3=total slices */
    int slice_mode_data;
    int disable_deblock;   /* 0=deblocking ON, 1=deblocking OFF */
    int    external_mem_fd;   /* -1 = use cuMemAlloc (legacy), >= 0 = import from Vulkan */
    size_t external_mem_size; /* Vulkan allocation size for import */
} nvenc_config_t;

/*
 * Initialise NVENC encoder for ARGB input with explicit config.
 *
 * Returns NULL on failure.
 */
nvenc_ctx_t *nvenc_layer_init_config(const nvenc_config_t *cfg);

/*
 * Initialise NVENC encoder for ARGB input (convenience wrapper).
 *
 * Parameters:
 *   width, height  - frame dimensions in pixels
 *   qp             - constant QP value (0-51)
 *
 * Encoder config: H.264 Baseline, CAVLC, const QP, deblocking ON,
 *   per-MB-row slices (slice_mode=3, slice_mode_data=height_mbs),
 *   outputAUD=1, maxNumRefFrames=1.
 *
 * Returns NULL on failure.
 */
nvenc_ctx_t *nvenc_layer_init(int width, int height, int qp);

/*
 * Encode one BGRA frame.
 *
 * Parameters:
 *   ctx          - encoder context from nvenc_layer_init()
 *   bgra_data    - pointer to BGRA pixel data (width * height * 4 bytes)
 *   bgra_size    - size of bgra_data in bytes
 *   force_idr    - if non-zero, force an IDR frame
 *   out_buf      - output buffer for the encoded bitstream
 *   out_buf_size - size of the output buffer
 *
 * Returns the number of bytes written to out_buf, or -1 on error.
 */
int nvenc_layer_encode(nvenc_ctx_t *ctx, const void *bgra_data, size_t bgra_size,
                       int force_idr, uint8_t *out_buf, size_t out_buf_size);

/*
 * Encode from GPU memory (zero-copy path).
 *
 * Assumes the Vulkan layer has already copied pixels into the device-local
 * buffer that backs this encoder's CUDA device pointer (via external memory).
 * Skips cuMemcpyHtoD entirely — just map → encode → lock → copy → unlock.
 *
 * Returns the number of bytes written to out_buf, or -1 on error.
 */
int nvenc_layer_encode_gpu(nvenc_ctx_t *ctx, int force_idr,
                           uint8_t *out_buf, size_t out_buf_size);

/*
 * Destroy the encoder and free all resources.
 */
void nvenc_layer_destroy(nvenc_ctx_t *ctx);

#endif /* NVENC_ENCODE_H */
