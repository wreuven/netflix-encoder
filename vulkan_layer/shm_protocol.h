/*
 * shm_protocol.h -- Shared memory protocol between VK_LAYER_CHROME_frame_capture
 *                   (C) and the Python frame analyzer.
 *
 * Layout in /dev/shm/chrome_frame_capture:
 *   [0 .. SHM_HEADER_SIZE)          shm_header_t  (control + metadata)
 *   [SHM_HEADER_SIZE .. SHM_TOTAL)  pixel data    (BGRA8, packed rows)
 */

#ifndef SHM_PROTOCOL_H
#define SHM_PROTOCOL_H

#include <stdint.h>

#define SHM_MAGIC          0x56464341  /* "VFCA" */
#define SHM_VERSION        4
#define SHM_NAME           "/chrome_frame_capture"

/* Maximum supported frame dimensions */
#define MAX_FRAME_WIDTH    3840
#define MAX_FRAME_HEIGHT   2160
#define BYTES_PER_PIXEL    4           /* BGRA8 */

/* Header is page-aligned for clean mmap boundary */
#define SHM_HEADER_SIZE    4096

/* Maximum pixel data size */
#define SHM_MAX_DATA_SIZE  ((uint64_t)MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * BYTES_PER_PIXEL)

/* Bitstream regions (after pixel data) for layer-side NVENC output */
#define SHM_BITSTREAM_OFFSET      (SHM_HEADER_SIZE + SHM_MAX_DATA_SIZE)
#define SHM_ENC1_BITSTREAM_OFFSET SHM_BITSTREAM_OFFSET
#define SHM_ENC1_BITSTREAM_MAX    (1024 * 1024)  /* 1MB for encoder1 */
#define SHM_ENC2_BITSTREAM_OFFSET (SHM_BITSTREAM_OFFSET + SHM_ENC1_BITSTREAM_MAX)
#define SHM_ENC2_BITSTREAM_MAX    (1024 * 1024)  /* 1MB for encoder2 */
#define SHM_BITSTREAM_MAX         (SHM_ENC1_BITSTREAM_MAX + SHM_ENC2_BITSTREAM_MAX)

/* Total shared memory region */
#define SHM_TOTAL_SIZE     (SHM_BITSTREAM_OFFSET + SHM_BITSTREAM_MAX)

/* Frame category constants (written by layer in frame_category field) */
#define FRAME_CAT_UNCHANGED   0
#define FRAME_CAT_VIDEO_ONLY  1
#define FRAME_CAT_CHANGED     2

/* Damage rect limits */
#define MAX_DAMAGE_RECTS        32
#define DAMAGE_RECT_NOT_PRESENT 0xFFFFFFFFu

typedef struct {
    int32_t  x;
    int32_t  y;
    uint32_t width;
    uint32_t height;
} shm_damage_rect_t;

/*
 * Control flags (atomic uint32_t).
 *
 *   Bit 0  LAYER_READY      Layer has initialised and is capturing.
 *   Bit 1  FRAME_AVAILABLE  A new frame is in the data region.
 *   Bit 2  CAPTURE_SUBRECT  If set, layer captures sub-rect; else full frame.
 *   Bit 3  READER_BUSY      Python is reading; layer must not overwrite.
 *   Bit 4  SHUTDOWN         Signal to layer to stop capturing.
 */
#define FLAG_LAYER_READY      (1u << 0)
#define FLAG_FRAME_AVAILABLE  (1u << 1)
#define FLAG_CAPTURE_SUBRECT  (1u << 2)
#define FLAG_READER_BUSY      (1u << 3)
#define FLAG_SHUTDOWN         (1u << 4)

typedef struct {
    /* Identification */
    uint32_t magic;                /* SHM_MAGIC                              */
    uint32_t version;              /* SHM_VERSION                            */
    uint32_t flags;                /* Atomic control flags                   */
    uint32_t _pad0;

    /* Swapchain / frame info (written by layer on swapchain create) */
    uint32_t frame_width;          /* Full swapchain width  (pixels)         */
    uint32_t frame_height;         /* Full swapchain height (pixels)         */
    uint32_t frame_format;         /* VkFormat enum value                    */
    uint32_t frame_stride;         /* Bytes per row for full frame           */

    /* Capture sub-rect request (written by Python) */
    int32_t  sub_x;                /* Left edge in pixels                    */
    int32_t  sub_y;                /* Top edge in pixels                     */
    uint32_t sub_width;            /* Width  (0 = full width)                */
    uint32_t sub_height;           /* Height (0 = full height)               */

    /* Actual captured region (written by layer after each copy) */
    int32_t  captured_x;
    int32_t  captured_y;
    uint32_t captured_width;
    uint32_t captured_height;
    uint32_t captured_stride;      /* Bytes per row in data region           */
    uint32_t _pad1;

    /* Sequence counters for lock-free synchronisation */
    uint64_t write_seq;            /* Incremented by layer after frame write */
    uint64_t read_seq;             /* Incremented by Python after frame read */

    /* Timing */
    uint64_t frame_timestamp_ns;   /* CLOCK_MONOTONIC nanoseconds            */
    uint32_t present_count;        /* Running count of presents seen         */
    uint32_t skip_count;           /* Presents where SHM write was skipped   */

    /* Damage rects from VK_KHR_incremental_present (offset 104) */
    uint32_t          damage_rect_count;  /* 0xFFFFFFFF = not available      */
    uint32_t          _pad2;
    shm_damage_rect_t damage_rects[MAX_DAMAGE_RECTS];  /* offset 112        */

    /* Encoder control (written by Python, offset 624) */
    uint32_t encode_request;       /* 0=pixels only, 1=encode+pixels, 2=encode+pixels+force_idr */
    uint32_t encoder_config;       /* QP in low 8 bits, reserved upper bits */

    /* Encoder output (written by layer, offset 632) */
    uint32_t bitstream_size;       /* Size of encoder1 bitstream in bytes, 0 if none */
    uint32_t _pad_enc;

    /* --- v4: Video state (written by Python, offset 640) --- */
    int32_t  video_rect_x;         /* device pixels (DPR+chrome adjusted) */
    int32_t  video_rect_y;
    uint32_t video_rect_w;
    uint32_t video_rect_h;
    uint64_t video_last_callback_ns; /* CLOCK_MONOTONIC ns of last video callback */
    uint32_t any_video_playing;    /* 1 = hero video on page */
    uint32_t _pad_video;
    /* 32 bytes: offsets 640-671 */

    /* --- v4: Classification + encoder2 output (written by layer, offset 672) --- */
    uint32_t frame_category;       /* FRAME_CAT_* */
    uint32_t encoder2_bitstream_size;
    int32_t  enc2_region_mb_x;    /* padded region in MB units */
    int32_t  enc2_region_mb_y;
    uint32_t enc2_region_mb_w;
    uint32_t enc2_region_mb_h;
    uint32_t _pad_cat[2];
    /* 32 bytes: offsets 672-703 */

    /* --- Reserved to pad header to exactly SHM_HEADER_SIZE --- */
    uint8_t  reserved[SHM_HEADER_SIZE - 112 - MAX_DAMAGE_RECTS * 16 - 16 - 64];
} shm_header_t;

_Static_assert(sizeof(shm_header_t) == SHM_HEADER_SIZE,
               "shm_header_t must be exactly SHM_HEADER_SIZE bytes");

#endif /* SHM_PROTOCOL_H */
