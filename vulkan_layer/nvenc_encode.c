/*
 * nvenc_encode.c -- NVENC encoder for the Vulkan capture layer.
 *
 * Pure C implementation using dlopen for CUDA and NVENC libraries.
 * Mirrors the approach in splicer/nvenc_lib.cpp but targets the layer's
 * specific requirements: ARGB input, per-MB-row slices, Baseline profile.
 */

#include "nvenc_encode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <nvEncodeAPI.h>

/* ================================================================== */
/* CUDA driver API types (minimal, loaded via dlopen)                   */
/* ================================================================== */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUdeviceptr;   /* actually unsigned long long on 64-bit */
#define CUDA_SUCCESS 0

typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*PFN_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (*PFN_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);

/* CUDA external memory (Vulkan interop) */
typedef CUresult (*PFN_cuImportExternalMemory)(void **, const void *);
typedef CUresult (*PFN_cuExternalMemoryGetMappedBuffer)(CUdeviceptr *, void *, const void *);
typedef CUresult (*PFN_cuDestroyExternalMemory)(void *);

#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD 1

typedef struct {
    uint64_t dummy[2];  /* reserved */
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_placeholder;

/*
 * We define the CUDA descriptor structs ourselves since we dlopen CUDA.
 * These must match the CUDA driver API ABI.
 */
typedef struct {
    int      handleType;          /* CUexternalMemoryHandleType */
    union {
        int  fd;
        struct { void *handle; const void *name; } win32;
        const void *nvSciBufObject;
    } handle;
    uint64_t size;
    unsigned int flags;
    unsigned int reserved[16];
} cuda_ext_mem_handle_desc_t;

typedef struct {
    uint64_t offset;
    uint64_t size;
    unsigned int flags;
    unsigned int reserved[16];
} cuda_ext_mem_buf_desc_t;

typedef NVENCSTATUS (NVENCAPI *PFN_NvEncodeAPICreateInstance)(NV_ENCODE_API_FUNCTION_LIST *);

/* ================================================================== */
/* Encoder context                                                      */
/* ================================================================== */

struct nvenc_ctx {
    /* Library handles */
    void *cuda_lib;
    void *nvenc_lib;

    /* CUDA function pointers */
    PFN_cuCtxDestroy  cuCtxDestroy;
    PFN_cuMemAlloc    cuMemAlloc;
    PFN_cuMemFree     cuMemFree;
    PFN_cuMemcpyHtoD  cuMemcpyHtoD;

    /* CUDA state */
    CUcontext   cu_ctx;
    CUdeviceptr dev_ptr;

    /* NVENC state */
    NV_ENCODE_API_FUNCTION_LIST api;
    void *encoder;
    void *registered_resource;
    void *bitstream_buffer;

    /* Frame parameters */
    int    width;
    int    height;
    int    pitch;        /* width * 4 for ARGB */
    size_t frame_size;   /* width * height * 4 */
    int    frame_count;

    /* External memory (Vulkan interop) */
    void *ext_mem;          /* CUexternalMemory handle, NULL when using cuMemAlloc */
    int   uses_external;    /* 1 = external memory from Vulkan fd */
    PFN_cuImportExternalMemory          cuImportExternalMemory;
    PFN_cuExternalMemoryGetMappedBuffer cuExternalMemoryGetMappedBuffer;
    PFN_cuDestroyExternalMemory         cuDestroyExternalMemory;
};

/* ================================================================== */
/* Init                                                                 */
/* ================================================================== */

nvenc_ctx_t *nvenc_layer_init_config(const nvenc_config_t *config)
{
    int width  = config->width;
    int height = config->height;
    int qp     = config->qp;

    nvenc_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;

    ctx->width      = width;
    ctx->height     = height;
    ctx->pitch      = width * 4;
    ctx->frame_size = (size_t)width * height * 4;

    /* --- Load CUDA --- */
    ctx->cuda_lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!ctx->cuda_lib) {
        fprintf(stderr, "[NVENC_LAYER] dlopen(libcuda.so.1): %s\n", dlerror());
        goto fail;
    }

    PFN_cuInit        fn_cuInit        = (PFN_cuInit)dlsym(ctx->cuda_lib, "cuInit");
    PFN_cuDeviceGet   fn_cuDeviceGet   = (PFN_cuDeviceGet)dlsym(ctx->cuda_lib, "cuDeviceGet");
    PFN_cuCtxCreate   fn_cuCtxCreate   = (PFN_cuCtxCreate)dlsym(ctx->cuda_lib, "cuCtxCreate_v2");
    ctx->cuCtxDestroy  = (PFN_cuCtxDestroy)dlsym(ctx->cuda_lib, "cuCtxDestroy_v2");
    ctx->cuMemAlloc    = (PFN_cuMemAlloc)dlsym(ctx->cuda_lib, "cuMemAlloc_v2");
    ctx->cuMemFree     = (PFN_cuMemFree)dlsym(ctx->cuda_lib, "cuMemFree_v2");
    ctx->cuMemcpyHtoD  = (PFN_cuMemcpyHtoD)dlsym(ctx->cuda_lib, "cuMemcpyHtoD_v2");

    if (!fn_cuInit || !fn_cuDeviceGet || !fn_cuCtxCreate ||
        !ctx->cuCtxDestroy || !ctx->cuMemAlloc || !ctx->cuMemFree ||
        !ctx->cuMemcpyHtoD) {
        fprintf(stderr, "[NVENC_LAYER] Failed to resolve CUDA symbols\n");
        goto fail;
    }

    if (fn_cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] cuInit failed\n");
        goto fail;
    }

    CUdevice cu_dev;
    if (fn_cuDeviceGet(&cu_dev, 0) != CUDA_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] cuDeviceGet failed\n");
        goto fail;
    }

    if (fn_cuCtxCreate(&ctx->cu_ctx, 0, cu_dev) != CUDA_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] cuCtxCreate failed\n");
        goto fail;
    }

    /* --- Load NVENC --- */
    ctx->nvenc_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!ctx->nvenc_lib) {
        fprintf(stderr, "[NVENC_LAYER] dlopen(libnvidia-encode.so.1): %s\n", dlerror());
        goto fail_cuda;
    }

    PFN_NvEncodeAPICreateInstance fn_create =
        (PFN_NvEncodeAPICreateInstance)dlsym(ctx->nvenc_lib, "NvEncodeAPICreateInstance");
    if (!fn_create) {
        fprintf(stderr, "[NVENC_LAYER] NvEncodeAPICreateInstance not found\n");
        goto fail_cuda;
    }

    ctx->api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (fn_create(&ctx->api) != NV_ENC_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] NvEncodeAPICreateInstance failed\n");
        goto fail_cuda;
    }

    /* --- Open encode session --- */
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sess = { 0 };
    sess.version    = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sess.device     = ctx->cu_ctx;
    sess.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sess.apiVersion = NVENCAPI_VERSION;

    NVENCSTATUS st = ctx->api.nvEncOpenEncodeSessionEx(&sess, &ctx->encoder);
    if (st != NV_ENC_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] nvEncOpenEncodeSessionEx failed: %d\n", st);
        goto fail_cuda;
    }

    /* --- Configure encoder --- */
    NV_ENC_PRESET_CONFIG preset_cfg = { 0 };
    preset_cfg.version = NV_ENC_PRESET_CONFIG_VER;
    preset_cfg.presetCfg.version = NV_ENC_CONFIG_VER;
    ctx->api.nvEncGetEncodePresetConfigEx(ctx->encoder, NV_ENC_CODEC_H264_GUID,
        NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_LOW_LATENCY, &preset_cfg);

    NV_ENC_INITIALIZE_PARAMS init = { 0 };
    init.version       = NV_ENC_INITIALIZE_PARAMS_VER;
    init.encodeGUID    = NV_ENC_CODEC_H264_GUID;
    init.presetGUID    = NV_ENC_PRESET_P4_GUID;
    init.encodeWidth   = (uint32_t)width;
    init.encodeHeight  = (uint32_t)height;
    init.darWidth      = (uint32_t)width;
    init.darHeight     = (uint32_t)height;
    init.frameRateNum  = 30;
    init.frameRateDen  = 1;
    init.enablePTD     = 1;
    init.encodeConfig  = &preset_cfg.presetCfg;
    init.tuningInfo    = NV_ENC_TUNING_INFO_LOW_LATENCY;

    NV_ENC_CONFIG *cfg = init.encodeConfig;
    cfg->profileGUID    = NV_ENC_H264_PROFILE_BASELINE_GUID;
    cfg->gopLength      = NVENC_INFINITE_GOPLENGTH;
    cfg->frameIntervalP = 1;  /* No B-frames */

    /* Constant QP */
    cfg->rcParams.rateControlMode     = NV_ENC_PARAMS_RC_CONSTQP;
    cfg->rcParams.constQP.qpInterP    = (uint32_t)qp;
    cfg->rcParams.constQP.qpInterB    = (uint32_t)qp;
    cfg->rcParams.constQP.qpIntra     = (uint32_t)qp;

    /* H.264 specifics — use config values */
    NV_ENC_CONFIG_H264 *h264 = &cfg->encodeCodecConfig.h264Config;
    h264->sliceMode                    = (uint32_t)config->slice_mode;
    h264->sliceModeData                = (uint32_t)config->slice_mode_data;
    h264->disableDeblockingFilterIDC   = (uint32_t)config->disable_deblock;
    h264->entropyCodingMode            = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
    h264->maxNumRefFrames              = 1;
    h264->outputAUD                    = 1;

    st = ctx->api.nvEncInitializeEncoder(ctx->encoder, &init);
    if (st != NV_ENC_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] nvEncInitializeEncoder failed: %d\n", st);
        ctx->api.nvEncDestroyEncoder(ctx->encoder);
        ctx->encoder = NULL;
        goto fail_cuda;
    }

    /* --- Allocate or import CUDA device memory for one ARGB frame --- */
    if (config->external_mem_fd >= 0) {
        /* Import Vulkan device-local memory via POSIX fd */
        ctx->cuImportExternalMemory =
            (PFN_cuImportExternalMemory)dlsym(ctx->cuda_lib, "cuImportExternalMemory");
        ctx->cuExternalMemoryGetMappedBuffer =
            (PFN_cuExternalMemoryGetMappedBuffer)dlsym(ctx->cuda_lib, "cuExternalMemoryGetMappedBuffer");
        ctx->cuDestroyExternalMemory =
            (PFN_cuDestroyExternalMemory)dlsym(ctx->cuda_lib, "cuDestroyExternalMemory");

        if (!ctx->cuImportExternalMemory || !ctx->cuExternalMemoryGetMappedBuffer ||
            !ctx->cuDestroyExternalMemory) {
            fprintf(stderr, "[NVENC_LAYER] Failed to resolve CUDA external memory symbols\n");
            ctx->api.nvEncDestroyEncoder(ctx->encoder);
            ctx->encoder = NULL;
            goto fail_cuda;
        }

        cuda_ext_mem_handle_desc_t hdesc;
        memset(&hdesc, 0, sizeof(hdesc));
        hdesc.handleType = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        hdesc.handle.fd  = config->external_mem_fd;
        hdesc.size       = config->external_mem_size;

        CUresult cr = ctx->cuImportExternalMemory(&ctx->ext_mem, &hdesc);
        if (cr != CUDA_SUCCESS) {
            fprintf(stderr, "[NVENC_LAYER] cuImportExternalMemory failed: %d\n", cr);
            ctx->api.nvEncDestroyEncoder(ctx->encoder);
            ctx->encoder = NULL;
            goto fail_cuda;
        }
        /* fd ownership transferred to CUDA, don't close it ourselves */

        cuda_ext_mem_buf_desc_t bdesc;
        memset(&bdesc, 0, sizeof(bdesc));
        bdesc.offset = 0;
        bdesc.size   = ctx->frame_size;

        cr = ctx->cuExternalMemoryGetMappedBuffer(&ctx->dev_ptr, ctx->ext_mem, &bdesc);
        if (cr != CUDA_SUCCESS) {
            fprintf(stderr, "[NVENC_LAYER] cuExternalMemoryGetMappedBuffer failed: %d\n", cr);
            ctx->cuDestroyExternalMemory(ctx->ext_mem);
            ctx->ext_mem = NULL;
            ctx->api.nvEncDestroyEncoder(ctx->encoder);
            ctx->encoder = NULL;
            goto fail_cuda;
        }

        ctx->uses_external = 1;
        fprintf(stderr, "[NVENC_LAYER] External memory imported: fd=%d, size=%zu, dev_ptr=%p\n",
                config->external_mem_fd, config->external_mem_size, (void *)ctx->dev_ptr);
    } else {
        /* Legacy path: allocate CUDA device memory */
        if (ctx->cuMemAlloc(&ctx->dev_ptr, ctx->frame_size) != CUDA_SUCCESS) {
            fprintf(stderr, "[NVENC_LAYER] cuMemAlloc failed for %zu bytes\n", ctx->frame_size);
            ctx->api.nvEncDestroyEncoder(ctx->encoder);
            ctx->encoder = NULL;
            goto fail_cuda;
        }
    }

    /* --- Register CUDA resource with NVENC --- */
    NV_ENC_REGISTER_RESOURCE reg = { 0 };
    reg.version            = NV_ENC_REGISTER_RESOURCE_VER;
    reg.resourceType       = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    reg.width              = (uint32_t)width;
    reg.height             = (uint32_t)height;
    reg.pitch              = (uint32_t)ctx->pitch;
    reg.resourceToRegister = (void *)ctx->dev_ptr;
    reg.bufferFormat       = NV_ENC_BUFFER_FORMAT_ARGB;
    reg.bufferUsage        = NV_ENC_INPUT_IMAGE;

    st = ctx->api.nvEncRegisterResource(ctx->encoder, &reg);
    if (st != NV_ENC_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] nvEncRegisterResource failed: %d\n", st);
        if (ctx->uses_external)
            ctx->cuDestroyExternalMemory(ctx->ext_mem);
        else
            ctx->cuMemFree(ctx->dev_ptr);
        ctx->api.nvEncDestroyEncoder(ctx->encoder);
        ctx->encoder = NULL;
        goto fail_cuda;
    }
    ctx->registered_resource = reg.registeredResource;

    /* --- Create bitstream output buffer --- */
    NV_ENC_CREATE_BITSTREAM_BUFFER bs = { 0 };
    bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    st = ctx->api.nvEncCreateBitstreamBuffer(ctx->encoder, &bs);
    if (st != NV_ENC_SUCCESS) {
        fprintf(stderr, "[NVENC_LAYER] nvEncCreateBitstreamBuffer failed: %d\n", st);
        ctx->api.nvEncUnregisterResource(ctx->encoder, ctx->registered_resource);
        if (ctx->uses_external)
            ctx->cuDestroyExternalMemory(ctx->ext_mem);
        else
            ctx->cuMemFree(ctx->dev_ptr);
        ctx->api.nvEncDestroyEncoder(ctx->encoder);
        ctx->encoder = NULL;
        goto fail_cuda;
    }
    ctx->bitstream_buffer = bs.bitstreamBuffer;

    fprintf(stderr, "[NVENC_LAYER] Encoder initialized: %dx%d, QP=%d, slice_mode=%d/%d, deblock=%s, ARGB, %s\n",
            width, height, qp, config->slice_mode, config->slice_mode_data,
            config->disable_deblock ? "OFF" : "ON",
            ctx->uses_external ? "external-mem" : "cuMemAlloc");

    return ctx;

fail_cuda:
    if (ctx->cu_ctx)
        ctx->cuCtxDestroy(ctx->cu_ctx);
fail:
    if (ctx->nvenc_lib)
        dlclose(ctx->nvenc_lib);
    if (ctx->cuda_lib)
        dlclose(ctx->cuda_lib);
    free(ctx);
    return NULL;
}

nvenc_ctx_t *nvenc_layer_init(int width, int height, int qp)
{
    int height_mbs = (height + 15) / 16;
    nvenc_config_t cfg = {
        .width           = width,
        .height          = height,
        .qp              = qp,
        .slice_mode      = 3,
        .slice_mode_data = height_mbs,
        .disable_deblock = 0,
        .external_mem_fd = -1,
    };
    return nvenc_layer_init_config(&cfg);
}

/* ================================================================== */
/* Encode                                                               */
/* ================================================================== */

int nvenc_layer_encode(nvenc_ctx_t *ctx, const void *bgra_data, size_t bgra_size,
                       int force_idr, uint8_t *out_buf, size_t out_buf_size)
{
    if (!ctx || !ctx->encoder)
        return -1;

    if (bgra_size < ctx->frame_size)
        return -1;

    /* Upload BGRA pixels to GPU */
    if (ctx->cuMemcpyHtoD(ctx->dev_ptr, bgra_data, ctx->frame_size) != CUDA_SUCCESS)
        return -1;

    /* Map registered resource */
    NV_ENC_MAP_INPUT_RESOURCE map = { 0 };
    map.version            = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map.registeredResource = ctx->registered_resource;

    NVENCSTATUS st = ctx->api.nvEncMapInputResource(ctx->encoder, &map);
    if (st != NV_ENC_SUCCESS)
        return -1;

    /* Encode */
    NV_ENC_PIC_PARAMS pic = { 0 };
    pic.version         = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth      = (uint32_t)ctx->width;
    pic.inputHeight     = (uint32_t)ctx->height;
    pic.inputPitch      = (uint32_t)ctx->pitch;
    pic.inputBuffer     = map.mappedResource;
    pic.outputBitstream = ctx->bitstream_buffer;
    pic.bufferFmt       = NV_ENC_BUFFER_FORMAT_ARGB;
    pic.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;

    if (force_idr || ctx->frame_count == 0)
        pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;

    st = ctx->api.nvEncEncodePicture(ctx->encoder, &pic);
    if (st != NV_ENC_SUCCESS) {
        ctx->api.nvEncUnmapInputResource(ctx->encoder, map.mappedResource);
        return -1;
    }

    /* Lock and copy bitstream */
    NV_ENC_LOCK_BITSTREAM lock = { 0 };
    lock.version         = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = ctx->bitstream_buffer;

    st = ctx->api.nvEncLockBitstream(ctx->encoder, &lock);
    if (st != NV_ENC_SUCCESS) {
        ctx->api.nvEncUnmapInputResource(ctx->encoder, map.mappedResource);
        return -1;
    }

    int result = -1;
    if (lock.bitstreamSizeInBytes <= out_buf_size) {
        memcpy(out_buf, lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes);
        result = (int)lock.bitstreamSizeInBytes;
    }

    ctx->api.nvEncUnlockBitstream(ctx->encoder, ctx->bitstream_buffer);
    ctx->api.nvEncUnmapInputResource(ctx->encoder, map.mappedResource);

    ctx->frame_count++;
    return result;
}

/* ================================================================== */
/* Encode from GPU memory (zero-copy)                                   */
/* ================================================================== */

int nvenc_layer_encode_gpu(nvenc_ctx_t *ctx, int force_idr,
                           uint8_t *out_buf, size_t out_buf_size)
{
    if (!ctx || !ctx->encoder)
        return -1;

    /* No cuMemcpyHtoD — pixels are already on-device via Vulkan external memory */

    /* Map registered resource */
    NV_ENC_MAP_INPUT_RESOURCE map = { 0 };
    map.version            = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map.registeredResource = ctx->registered_resource;

    NVENCSTATUS st = ctx->api.nvEncMapInputResource(ctx->encoder, &map);
    if (st != NV_ENC_SUCCESS)
        return -1;

    /* Encode */
    NV_ENC_PIC_PARAMS pic = { 0 };
    pic.version         = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth      = (uint32_t)ctx->width;
    pic.inputHeight     = (uint32_t)ctx->height;
    pic.inputPitch      = (uint32_t)ctx->pitch;
    pic.inputBuffer     = map.mappedResource;
    pic.outputBitstream = ctx->bitstream_buffer;
    pic.bufferFmt       = NV_ENC_BUFFER_FORMAT_ARGB;
    pic.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;

    if (force_idr || ctx->frame_count == 0)
        pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;

    st = ctx->api.nvEncEncodePicture(ctx->encoder, &pic);
    if (st != NV_ENC_SUCCESS) {
        ctx->api.nvEncUnmapInputResource(ctx->encoder, map.mappedResource);
        return -1;
    }

    /* Lock and copy bitstream */
    NV_ENC_LOCK_BITSTREAM lock = { 0 };
    lock.version         = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = ctx->bitstream_buffer;

    st = ctx->api.nvEncLockBitstream(ctx->encoder, &lock);
    if (st != NV_ENC_SUCCESS) {
        ctx->api.nvEncUnmapInputResource(ctx->encoder, map.mappedResource);
        return -1;
    }

    int result = -1;
    if (lock.bitstreamSizeInBytes <= out_buf_size) {
        memcpy(out_buf, lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes);
        result = (int)lock.bitstreamSizeInBytes;
    }

    ctx->api.nvEncUnlockBitstream(ctx->encoder, ctx->bitstream_buffer);
    ctx->api.nvEncUnmapInputResource(ctx->encoder, map.mappedResource);

    ctx->frame_count++;
    return result;
}

/* ================================================================== */
/* Destroy                                                              */
/* ================================================================== */

void nvenc_layer_destroy(nvenc_ctx_t *ctx)
{
    if (!ctx)
        return;

    if (ctx->encoder) {
        if (ctx->bitstream_buffer)
            ctx->api.nvEncDestroyBitstreamBuffer(ctx->encoder, ctx->bitstream_buffer);
        if (ctx->registered_resource)
            ctx->api.nvEncUnregisterResource(ctx->encoder, ctx->registered_resource);
        ctx->api.nvEncDestroyEncoder(ctx->encoder);
    }

    if (ctx->uses_external) {
        if (ctx->ext_mem)
            ctx->cuDestroyExternalMemory(ctx->ext_mem);
        /* dev_ptr is a mapping from external memory, not a cuMemAlloc — don't cuMemFree */
    } else {
        if (ctx->dev_ptr)
            ctx->cuMemFree(ctx->dev_ptr);
    }
    if (ctx->cu_ctx)
        ctx->cuCtxDestroy(ctx->cu_ctx);
    if (ctx->nvenc_lib)
        dlclose(ctx->nvenc_lib);
    if (ctx->cuda_lib)
        dlclose(ctx->cuda_lib);

    free(ctx);
}
