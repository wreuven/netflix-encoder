/*
 * frame_capture_layer.c -- VK_LAYER_CHROME_frame_capture
 *
 * A Vulkan implicit layer that intercepts vkQueuePresentKHR to copy the
 * presented swapchain image into POSIX shared memory for analysis by an
 * external Python process.
 *
 * Activation:  set  CHROME_FRAME_CAPTURE=1  in the environment.
 * Layer path:  set  VK_LAYER_PATH  to the directory containing this .so
 *              and the accompanying .json manifest.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "frame_capture_layer.h"
#include "shm_protocol.h"

/* ================================================================== */
/* Global tables                                                       */
/* ================================================================== */

static fc_instance_data_t *g_instances[MAX_INSTANCES];
static fc_device_data_t   *g_devices[MAX_DEVICES];
static pthread_mutex_t     g_lock = PTHREAD_MUTEX_INITIALIZER;

/* ================================================================== */
/* Table helpers                                                       */
/* ================================================================== */

static fc_instance_data_t *find_instance(VkInstance instance) {
    void *key = GET_DISPATCH_KEY(instance);
    for (int i = 0; i < MAX_INSTANCES; i++)
        if (g_instances[i] && g_instances[i]->dispatch_key == key)
            return g_instances[i];
    return NULL;
}

static fc_device_data_t *find_device(VkDevice device) {
    void *key = GET_DISPATCH_KEY(device);
    for (int i = 0; i < MAX_DEVICES; i++)
        if (g_devices[i] && g_devices[i]->dispatch_key == key)
            return g_devices[i];
    return NULL;
}

static fc_device_data_t *find_device_by_queue(VkQueue queue) {
    void *key = GET_DISPATCH_KEY(queue);
    for (int i = 0; i < MAX_DEVICES; i++)
        if (g_devices[i] && g_devices[i]->dispatch_key == key)
            return g_devices[i];
    return NULL;
}

static void store_instance(fc_instance_data_t *data) {
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_INSTANCES; i++) {
        if (!g_instances[i]) { g_instances[i] = data; break; }
    }
    pthread_mutex_unlock(&g_lock);
}

static void remove_instance(VkInstance instance) {
    void *key = GET_DISPATCH_KEY(instance);
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_INSTANCES; i++) {
        if (g_instances[i] && g_instances[i]->dispatch_key == key) {
            free(g_instances[i]);
            g_instances[i] = NULL;
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
}

static void store_device(fc_device_data_t *data) {
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (!g_devices[i]) { g_devices[i] = data; break; }
    }
    pthread_mutex_unlock(&g_lock);
}

static void remove_device(VkDevice device) {
    void *key = GET_DISPATCH_KEY(device);
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (g_devices[i] && g_devices[i]->dispatch_key == key) {
            /* Caller frees the struct */
            g_devices[i] = NULL;
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
}

/* ================================================================== */
/* Timing                                                              */
/* ================================================================== */

static uint64_t get_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ================================================================== */
/* Damage rect helpers                                                 */
/* ================================================================== */

static const VkPresentRegionsKHR *
find_present_regions(const VkPresentInfoKHR *info)
{
    const VkBaseInStructure *s = (const VkBaseInStructure *)info->pNext;
    while (s) {
        if (s->sType == VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR)
            return (const VkPresentRegionsKHR *)s;
        s = s->pNext;
    }
    return NULL;
}

static void accumulate_damage(fc_device_data_t *d,
                              const VkPresentInfoKHR *pPresentInfo)
{
    const VkPresentRegionsKHR *regions = find_present_regions(pPresentInfo);
    if (!regions) {
        /* Extension not present — leave sentinel so Python knows */
        return;
    }

    /* First time we see damage rects: flip sentinel to 0 */
    if (d->pending_damage_count == DAMAGE_RECT_NOT_PRESENT)
        d->pending_damage_count = 0;

    if (d->damage_overflow)
        return;  /* already overflowed, will flush as full-frame */

    /* Use first swapchain's regions */
    if (regions->swapchainCount == 0 || !regions->pRegions)
        return;

    const VkPresentRegionKHR *region = &regions->pRegions[0];

    if (region->rectangleCount == 0) {
        /* Vulkan spec: 0 rectangles = whole surface damaged */
        if (d->pending_damage_count >= MAX_DAMAGE_RECTS) {
            d->damage_overflow = 1;
            return;
        }
        shm_damage_rect_t *r = &d->pending_damage[d->pending_damage_count++];
        r->x = 0;
        r->y = 0;
        r->width  = d->swapchain_extent.width;
        r->height = d->swapchain_extent.height;
        return;
    }

    for (uint32_t i = 0; i < region->rectangleCount; i++) {
        if (d->pending_damage_count >= MAX_DAMAGE_RECTS) {
            d->damage_overflow = 1;
            return;
        }
        const VkRectLayerKHR *vr = &region->pRectangles[i];
        shm_damage_rect_t *r = &d->pending_damage[d->pending_damage_count++];
        r->x      = vr->offset.x;
        r->y      = vr->offset.y;
        r->width  = vr->extent.width;
        r->height = vr->extent.height;
    }
}

static void flush_damage_to_shm(fc_device_data_t *d)
{
    shm_header_t *hdr = d->shm_header;
    if (!hdr)
        return;

    if (d->pending_damage_count == DAMAGE_RECT_NOT_PRESENT) {
        /* Extension never seen — write sentinel */
        hdr->damage_rect_count = DAMAGE_RECT_NOT_PRESENT;
    } else if (d->damage_overflow) {
        /* Overflow: report one full-frame rect */
        hdr->damage_rect_count = 1;
        hdr->damage_rects[0].x = 0;
        hdr->damage_rects[0].y = 0;
        hdr->damage_rects[0].width  = d->swapchain_extent.width;
        hdr->damage_rects[0].height = d->swapchain_extent.height;
    } else {
        hdr->damage_rect_count = d->pending_damage_count;
        if (d->pending_damage_count > 0) {
            memcpy(hdr->damage_rects, d->pending_damage,
                   d->pending_damage_count * sizeof(shm_damage_rect_t));
        }
    }

    /* Reset accumulator */
    d->pending_damage_count = DAMAGE_RECT_NOT_PRESENT;
    d->damage_overflow = 0;
}

/* ================================================================== */
/* Shared memory                                                       */
/* ================================================================== */

static void init_shm(fc_device_data_t *d) {
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        fprintf(stderr, "[FC_LAYER] shm_open(%s): %s\n", SHM_NAME, strerror(errno));
        return;
    }
    if (ftruncate(fd, (off_t)SHM_TOTAL_SIZE) < 0) {
        fprintf(stderr, "[FC_LAYER] ftruncate: %s\n", strerror(errno));
        close(fd);
        return;
    }
    void *ptr = mmap(NULL, (size_t)SHM_TOTAL_SIZE, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "[FC_LAYER] mmap: %s\n", strerror(errno));
        close(fd);
        return;
    }

    d->shm_fd     = fd;
    d->shm_header = (shm_header_t *)ptr;
    d->shm_data   = (uint8_t *)ptr + SHM_HEADER_SIZE;

    memset(d->shm_header, 0, SHM_HEADER_SIZE);
    d->shm_header->magic   = SHM_MAGIC;
    d->shm_header->version = SHM_VERSION;
    d->shm_header->damage_rect_count = DAMAGE_RECT_NOT_PRESENT;

    fprintf(stderr, "[FC_LAYER] Shared memory initialised (%s, %lu bytes)\n",
            SHM_NAME, (unsigned long)SHM_TOTAL_SIZE);
}

static void destroy_shm(fc_device_data_t *d) {
    if (d->shm_header) {
        munmap(d->shm_header, (size_t)SHM_TOTAL_SIZE);
        d->shm_header = NULL;
        d->shm_data   = NULL;
    }
    if (d->shm_fd >= 0) {
        close(d->shm_fd);
        shm_unlink(SHM_NAME);
        d->shm_fd = -1;
    }
}

/* ================================================================== */
/* Staging buffer + command resources                                   */
/* ================================================================== */

static uint32_t find_memory_type(fc_device_data_t *d,
                                 uint32_t type_bits,
                                 VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties props;
    d->fpGetPhysicalDeviceMemoryProperties(d->physical_device, &props);

    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return UINT32_MAX;
}

static void create_staging(fc_device_data_t *d) {
    VkDeviceSize size = (VkDeviceSize)d->swapchain_extent.width *
                        d->swapchain_extent.height * BYTES_PER_PIXEL;

    VkBufferCreateInfo buf_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (d->fpCreateBuffer(d->device, &buf_ci, NULL, &d->staging_buf) != VK_SUCCESS) {
        fprintf(stderr, "[FC_LAYER] Failed to create staging buffer\n");
        return;
    }

    VkMemoryRequirements mem_req;
    d->fpGetBufferMemoryRequirements(d->device, d->staging_buf, &mem_req);

    uint32_t mem_type = find_memory_type(d, mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type == UINT32_MAX) {
        fprintf(stderr, "[FC_LAYER] No host-visible coherent memory type\n");
        return;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = mem_req.size,
        .memoryTypeIndex = mem_type,
    };
    if (d->fpAllocateMemory(d->device, &alloc_info, NULL, &d->staging_mem) != VK_SUCCESS) {
        fprintf(stderr, "[FC_LAYER] Failed to allocate staging memory\n");
        return;
    }
    d->fpBindBufferMemory(d->device, d->staging_buf, d->staging_mem, 0);
    d->fpMapMemory(d->device, d->staging_mem, 0, size, 0, &d->staging_mapped);
    d->staging_size = size;

    /* Command pool */
    VkCommandPoolCreateInfo pool_ci = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = d->queue_family,
    };
    d->fpCreateCommandPool(d->device, &pool_ci, NULL, &d->cmd_pool);

    VkCommandBufferAllocateInfo cb_ai = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = d->cmd_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    d->fpAllocateCommandBuffers(d->device, &cb_ai, &d->cmd_buf);

    VkFenceCreateInfo fence_ci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    d->fpCreateFence(d->device, &fence_ci, NULL, &d->fence);

    fprintf(stderr, "[FC_LAYER] Staging resources created (%ux%u, %lu bytes)\n",
            d->swapchain_extent.width, d->swapchain_extent.height,
            (unsigned long)size);
}

static void destroy_staging(fc_device_data_t *d) {
    if (d->nvenc_enc1) {
        nvenc_layer_destroy(d->nvenc_enc1);
        d->nvenc_enc1 = NULL;
    }
    if (d->nvenc_enc2) {
        nvenc_layer_destroy(d->nvenc_enc2);
        d->nvenc_enc2 = NULL;
        d->enc2_width = 0;
        d->enc2_height = 0;
    }
    if (d->fence) {
        d->fpDestroyFence(d->device, d->fence, NULL);
        d->fence = VK_NULL_HANDLE;
    }
    if (d->cmd_buf) {
        d->fpFreeCommandBuffers(d->device, d->cmd_pool, 1, &d->cmd_buf);
        d->cmd_buf = VK_NULL_HANDLE;
    }
    if (d->cmd_pool) {
        d->fpDestroyCommandPool(d->device, d->cmd_pool, NULL);
        d->cmd_pool = VK_NULL_HANDLE;
    }
    if (d->staging_mapped) {
        d->fpUnmapMemory(d->device, d->staging_mem);
        d->staging_mapped = NULL;
    }
    if (d->staging_mem) {
        d->fpFreeMemory(d->device, d->staging_mem, NULL);
        d->staging_mem = VK_NULL_HANDLE;
    }
    if (d->staging_buf) {
        d->fpDestroyBuffer(d->device, d->staging_buf, NULL);
        d->staging_buf = VK_NULL_HANDLE;
    }
}

/* ================================================================== */
/* Frame classification (v4)                                           */
/* ================================================================== */

#define VIDEO_CALLBACK_WINDOW_NS  200000000ULL  /* 200ms */
#define DAMAGE_TOLERANCE          4             /* pixels */

static int classify_frame(fc_device_data_t *d)
{
    shm_header_t *hdr = d->shm_header;

    /* Read damage rects from header (already flushed by caller) */
    uint32_t damage_count = hdr->damage_rect_count;
    int has_damage = (damage_count != DAMAGE_RECT_NOT_PRESENT && damage_count > 0);

    /* Read video state written by Python */
    uint64_t now_ns = get_monotonic_ns();
    uint64_t video_cb_ns = hdr->video_last_callback_ns;
    int video_rect_valid = (hdr->video_rect_w > 0 && hdr->video_rect_h > 0);
    int recent_video_cb = (video_cb_ns > 0 && (now_ns - video_cb_ns) < VIDEO_CALLBACK_WINDOW_NS);
    int any_playing = hdr->any_video_playing;

    if (!has_damage) {
        /* No damage rects */
        if (recent_video_cb)
            return FRAME_CAT_VIDEO_ONLY;
        if (any_playing)
            return FRAME_CAT_CHANGED;  /* hero video */
        return FRAME_CAT_UNCHANGED;
    }

    /* Non-empty damage rects — check if all within video_rect ± tolerance */
    if (video_rect_valid && recent_video_cb) {
        int32_t vx = hdr->video_rect_x - DAMAGE_TOLERANCE;
        int32_t vy = hdr->video_rect_y - DAMAGE_TOLERANCE;
        int32_t vr = hdr->video_rect_x + (int32_t)hdr->video_rect_w + DAMAGE_TOLERANCE;
        int32_t vb = hdr->video_rect_y + (int32_t)hdr->video_rect_h + DAMAGE_TOLERANCE;

        int all_within = 1;
        uint32_t n = (damage_count > MAX_DAMAGE_RECTS) ? MAX_DAMAGE_RECTS : damage_count;
        for (uint32_t i = 0; i < n; i++) {
            const shm_damage_rect_t *dr = &hdr->damage_rects[i];
            if (dr->x < vx || dr->y < vy ||
                dr->x + (int32_t)dr->width > vr ||
                dr->y + (int32_t)dr->height > vb) {
                all_within = 0;
                break;
            }
        }
        if (all_within)
            return FRAME_CAT_VIDEO_ONLY;
    }

    return FRAME_CAT_CHANGED;
}

/* ================================================================== */
/* Frame capture (called from vkQueuePresentKHR)                       */
/* ================================================================== */

static void capture_frame(fc_device_data_t *d, VkQueue queue,
                          VkImage src_image) {
    shm_header_t *hdr = d->shm_header;
    if (!hdr || !d->staging_mapped)
        return;

    /* Check shutdown flag */
    uint32_t flags = __atomic_load_n(&hdr->flags, __ATOMIC_ACQUIRE);
    if (flags & FLAG_SHUTDOWN)
        return;

    /* Don't write if reader is busy or hasn't consumed the last frame */
    if (flags & FLAG_READER_BUSY) {
        hdr->skip_count++;
        return;
    }
    if (hdr->write_seq > 0 && hdr->read_seq < hdr->write_seq) {
        hdr->skip_count++;
        return;
    }

    /* Always capture full frame for classification + encoding */
    uint32_t cap_w = d->swapchain_extent.width;
    uint32_t cap_h = d->swapchain_extent.height;

    /* Check that the copy fits in staging buffer */
    VkDeviceSize copy_size = (VkDeviceSize)cap_w * cap_h * BYTES_PER_PIXEL;
    if (copy_size > d->staging_size)
        return;

    /* --- Record command buffer --- */
    d->fpResetCommandBuffer(d->cmd_buf, 0);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    d->fpBeginCommandBuffer(d->cmd_buf, &begin_info);

    /* Barrier: PRESENT_SRC -> TRANSFER_SRC */
    VkImageMemoryBarrier barrier_to_src = {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT,
        .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = src_image,
        .subresourceRange    = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };
    d->fpCmdPipelineBarrier(d->cmd_buf,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier_to_src);

    /* Copy full frame to staging buffer */
    VkBufferImageCopy region = {
        .bufferOffset      = 0,
        .bufferRowLength   = cap_w,
        .bufferImageHeight = cap_h,
        .imageSubresource  = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel       = 0,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { cap_w, cap_h, 1 },
    };
    d->fpCmdCopyImageToBuffer(d->cmd_buf, src_image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        d->staging_buf, 1, &region);

    /* Barrier: TRANSFER_SRC -> PRESENT_SRC */
    VkImageMemoryBarrier barrier_to_present = barrier_to_src;
    barrier_to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier_to_present.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    barrier_to_present.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier_to_present.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    d->fpCmdPipelineBarrier(d->cmd_buf,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier_to_present);

    d->fpEndCommandBuffer(d->cmd_buf);

    /* Submit and wait */
    VkSubmitInfo submit_info = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &d->cmd_buf,
    };
    d->fpResetFences(d->device, 1, &d->fence);
    d->fpQueueSubmit(queue, 1, &submit_info, d->fence);
    d->fpWaitForFences(d->device, 1, &d->fence, VK_TRUE, UINT64_MAX);

    /* --- Flush damage rects before classification --- */
    flush_damage_to_shm(d);

    /* --- Classify frame --- */
    int category = classify_frame(d);

    /* --- Defaults --- */
    uint32_t row_bytes = cap_w * BYTES_PER_PIXEL;
    int src_stride = (int)(cap_w * BYTES_PER_PIXEL);
    hdr->bitstream_size = 0;
    hdr->encoder2_bitstream_size = 0;
    hdr->enc2_region_mb_x = 0;
    hdr->enc2_region_mb_y = 0;
    hdr->enc2_region_mb_w = 0;
    hdr->enc2_region_mb_h = 0;

    int qp = hdr->encoder_config & 0xFF;
    if (qp == 0) qp = 23;

    if (category == FRAME_CAT_CHANGED) {
        /* --- Encode full frame with encoder1 --- */
        if (!d->nvenc_enc1) {
            d->nvenc_enc1 = nvenc_layer_init((int)cap_w, (int)cap_h, qp);
            if (d->nvenc_enc1)
                fprintf(stderr, "[FC_LAYER] NVENC encoder1 (full-frame) initialized: %ux%u\n",
                        cap_w, cap_h);
            else
                fprintf(stderr, "[FC_LAYER] NVENC encoder1 init FAILED\n");
        }

        if (d->nvenc_enc1) {
            VkDeviceSize px_size = (VkDeviceSize)cap_h * row_bytes;
            uint8_t *bs_out = (uint8_t *)hdr + SHM_ENC1_BITSTREAM_OFFSET;
            int bs_size = nvenc_layer_encode(d->nvenc_enc1, d->staging_mapped,
                                             (size_t)px_size, 0,
                                             bs_out, SHM_ENC1_BITSTREAM_MAX);
            hdr->bitstream_size = (bs_size > 0) ? (uint32_t)bs_size : 0;
        }

        d->enc2_needs_idr = 1;

    } else if (category == FRAME_CAT_VIDEO_ONLY) {
        /* --- Compute padded region (video_rect + 1 MB border, MB-aligned) --- */
        int padding_px = 16;  /* 1 MB border */
        int vx = hdr->video_rect_x;
        int vy = hdr->video_rect_y;
        int vw = (int)hdr->video_rect_w;
        int vh = (int)hdr->video_rect_h;

        int padded_x = ((vx / 16) * 16 - padding_px);
        int padded_y = ((vy / 16) * 16 - padding_px);
        if (padded_x < 0) padded_x = 0;
        if (padded_y < 0) padded_y = 0;

        int padded_x2 = (((vx + vw + 15) / 16) * 16 + padding_px);
        int padded_y2 = (((vy + vh + 15) / 16) * 16 + padding_px);
        if (padded_x2 > (int)cap_w) padded_x2 = (int)cap_w;
        if (padded_y2 > (int)cap_h) padded_y2 = (int)cap_h;

        int padded_w = ((padded_x2 - padded_x) / 16) * 16;
        int padded_h = ((padded_y2 - padded_y) / 16) * 16;

        if (padded_w >= 32 && padded_h >= 32) {
            int size_changed = (padded_w != d->enc2_width || padded_h != d->enc2_height);

            /* Reinit encoder2 if region size changed */
            if (d->nvenc_enc2 && size_changed) {
                nvenc_layer_destroy(d->nvenc_enc2);
                d->nvenc_enc2 = NULL;
            }

            if (!d->nvenc_enc2) {
                int height_mbs = (padded_h + 15) / 16;
                nvenc_config_t enc2_cfg = {
                    .width          = padded_w,
                    .height         = padded_h,
                    .qp             = qp,
                    .slice_mode     = 2,     /* MB rows per slice */
                    .slice_mode_data = 1,    /* 1 MB row per slice */
                    .disable_deblock = 1,    /* deblocking OFF for splicing */
                };
                (void)height_mbs;
                d->nvenc_enc2 = nvenc_layer_init_config(&enc2_cfg);
                if (d->nvenc_enc2) {
                    d->enc2_width = padded_w;
                    d->enc2_height = padded_h;
                    d->enc2_needs_idr = 1;  /* new encoder always starts with IDR */
                    fprintf(stderr, "[FC_LAYER] NVENC encoder2 (region) initialized: %dx%d\n",
                            padded_w, padded_h);
                } else {
                    fprintf(stderr, "[FC_LAYER] NVENC encoder2 init FAILED\n");
                }
            }

            if (d->nvenc_enc2) {
                int force_idr = d->enc2_needs_idr || size_changed;
                uint8_t *bs_out = (uint8_t *)hdr + SHM_ENC2_BITSTREAM_OFFSET;
                int bs_size = nvenc_layer_encode_region(d->nvenc_enc2,
                    d->staging_mapped, src_stride,
                    padded_x, padded_y, padded_w, padded_h,
                    force_idr, bs_out, SHM_ENC2_BITSTREAM_MAX);
                hdr->encoder2_bitstream_size = (bs_size > 0) ? (uint32_t)bs_size : 0;
                hdr->enc2_region_mb_x = padded_x / 16;
                hdr->enc2_region_mb_y = padded_y / 16;
                hdr->enc2_region_mb_w = padded_w / 16;
                hdr->enc2_region_mb_h = padded_h / 16;
                d->enc2_needs_idr = 0;
            }
        }
        /* else: region too small, category stays VIDEO_ONLY but no bitstream */
    }
    /* FRAME_CAT_UNCHANGED: no encoding, no pixel copy */

    /* --- Optional debug pixel copy --- */
    if (flags & FLAG_COPY_PIXELS) {
        VkDeviceSize px_size = (VkDeviceSize)cap_h * row_bytes;
        memcpy(d->shm_data, d->staging_mapped, (size_t)px_size);
    }

    /* Update header */
    hdr->frame_category  = (uint32_t)category;
    hdr->captured_x      = 0;
    hdr->captured_y      = 0;
    hdr->captured_width  = cap_w;
    hdr->captured_height = cap_h;
    hdr->captured_stride = row_bytes;
    hdr->frame_timestamp_ns = get_monotonic_ns();
    hdr->present_count++;

    __atomic_add_fetch(&hdr->write_seq, 1, __ATOMIC_SEQ_CST);
    __atomic_or_fetch(&hdr->flags, FLAG_FRAME_AVAILABLE, __ATOMIC_SEQ_CST);
}

/* ================================================================== */
/* Intercepted Vulkan functions                                        */
/* ================================================================== */

/* ---- vkCreateInstance ---- */

static VKAPI_ATTR VkResult VKAPI_CALL
FC_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator,
                  VkInstance *pInstance)
{
    /* Walk the pNext chain to find the layer chain info */
    VkLayerInstanceCreateInfo *chain_info = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (chain_info &&
           !(chain_info->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
             chain_info->function == VK_LAYER_LINK_INFO))
        chain_info = (VkLayerInstanceCreateInfo *)chain_info->pNext;

    if (!chain_info)
        return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr =
        chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;

    /* Advance the chain for the next layer */
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateInstance fpCreateInstance =
        (PFN_vkCreateInstance)fpGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");

    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS)
        return result;

    fc_instance_data_t *data = calloc(1, sizeof(*data));
    data->dispatch_key = GET_DISPATCH_KEY(*pInstance);
    data->instance     = *pInstance;
    data->fpGetInstanceProcAddr = fpGetInstanceProcAddr;
    data->fpDestroyInstance =
        (PFN_vkDestroyInstance)fpGetInstanceProcAddr(*pInstance, "vkDestroyInstance");
    data->fpEnumerateDeviceExtensionProperties =
        (PFN_vkEnumerateDeviceExtensionProperties)fpGetInstanceProcAddr(
            *pInstance, "vkEnumerateDeviceExtensionProperties");
    data->fpGetPhysicalDeviceMemoryProperties =
        (PFN_vkGetPhysicalDeviceMemoryProperties)fpGetInstanceProcAddr(
            *pInstance, "vkGetPhysicalDeviceMemoryProperties");

    store_instance(data);

    fprintf(stderr, "[FC_LAYER] Instance created\n");
    return VK_SUCCESS;
}

/* ---- vkDestroyInstance ---- */

static VKAPI_ATTR void VKAPI_CALL
FC_DestroyInstance(VkInstance instance,
                   const VkAllocationCallbacks *pAllocator)
{
    fc_instance_data_t *data = find_instance(instance);
    if (data) {
        data->fpDestroyInstance(instance, pAllocator);
        remove_instance(instance);
    }
    fprintf(stderr, "[FC_LAYER] Instance destroyed\n");
}

/* ---- vkCreateDevice ---- */

static VKAPI_ATTR VkResult VKAPI_CALL
FC_CreateDevice(VkPhysicalDevice physicalDevice,
                const VkDeviceCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator,
                VkDevice *pDevice)
{
    /* Walk pNext chain for layer link info */
    VkLayerDeviceCreateInfo *chain_info = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (chain_info &&
           !(chain_info->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
             chain_info->function == VK_LAYER_LINK_INFO))
        chain_info = (VkLayerDeviceCreateInfo *)chain_info->pNext;

    if (!chain_info)
        return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr =
        chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr =
        chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;

    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateDevice fpCreateDevice =
        (PFN_vkCreateDevice)fpGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice");

    VkResult result = fpCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS)
        return result;

    fc_device_data_t *d = calloc(1, sizeof(*d));
    d->dispatch_key    = GET_DISPATCH_KEY(*pDevice);
    d->device          = *pDevice;
    d->physical_device = physicalDevice;
    d->shm_fd          = -1;

    /* Use first queue family that supports graphics or transfer */
    d->queue_family = pCreateInfo->pQueueCreateInfos[0].queueFamilyIndex;

    /* Resolve all device-level function pointers */
#define RESOLVE(name) d->fp##name = (PFN_vk##name)fpGetDeviceProcAddr(*pDevice, "vk" #name)
    d->fpGetDeviceProcAddr = fpGetDeviceProcAddr;
    RESOLVE(DestroyDevice);
    RESOLVE(CreateSwapchainKHR);
    RESOLVE(DestroySwapchainKHR);
    RESOLVE(GetSwapchainImagesKHR);
    RESOLVE(QueuePresentKHR);
    RESOLVE(QueueSubmit);
    RESOLVE(QueueWaitIdle);
    RESOLVE(CreateCommandPool);
    RESOLVE(DestroyCommandPool);
    RESOLVE(AllocateCommandBuffers);
    RESOLVE(FreeCommandBuffers);
    RESOLVE(BeginCommandBuffer);
    RESOLVE(EndCommandBuffer);
    RESOLVE(ResetCommandBuffer);
    RESOLVE(CmdPipelineBarrier);
    RESOLVE(CmdCopyImageToBuffer);
    RESOLVE(CreateBuffer);
    RESOLVE(DestroyBuffer);
    RESOLVE(GetBufferMemoryRequirements);
    RESOLVE(AllocateMemory);
    RESOLVE(FreeMemory);
    RESOLVE(BindBufferMemory);
    RESOLVE(MapMemory);
    RESOLVE(UnmapMemory);
    RESOLVE(CreateFence);
    RESOLVE(DestroyFence);
    RESOLVE(WaitForFences);
    RESOLVE(ResetFences);
#undef RESOLVE

    /* Resolve physical device memory properties via instance */
    fc_instance_data_t *inst = NULL;
    for (int i = 0; i < MAX_INSTANCES; i++) {
        if (g_instances[i]) { inst = g_instances[i]; break; }
    }
    if (inst)
        d->fpGetPhysicalDeviceMemoryProperties = inst->fpGetPhysicalDeviceMemoryProperties;

    /* Initialise shared memory */
    init_shm(d);

    store_device(d);

    fprintf(stderr, "[FC_LAYER] Device created (queue family %u)\n", d->queue_family);
    return VK_SUCCESS;
}

/* ---- vkDestroyDevice ---- */

static VKAPI_ATTR void VKAPI_CALL
FC_DestroyDevice(VkDevice device,
                 const VkAllocationCallbacks *pAllocator)
{
    fc_device_data_t *d = find_device(device);
    if (!d) return;

    destroy_staging(d);
    destroy_shm(d);
    remove_device(device);

    d->fpDestroyDevice(device, pAllocator);
    free(d->swapchain_images);
    free(d);

    fprintf(stderr, "[FC_LAYER] Device destroyed\n");
}

/* ---- vkCreateSwapchainKHR ---- */

static VKAPI_ATTR VkResult VKAPI_CALL
FC_CreateSwapchainKHR(VkDevice device,
                      const VkSwapchainCreateInfoKHR *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkSwapchainKHR *pSwapchain)
{
    fc_device_data_t *d = find_device(device);
    if (!d) return VK_ERROR_DEVICE_LOST;

    /* Add TRANSFER_SRC so we can copy from swapchain images */
    VkSwapchainCreateInfoKHR modified = *pCreateInfo;
    modified.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkResult result = d->fpCreateSwapchainKHR(device, &modified, pAllocator, pSwapchain);
    if (result != VK_SUCCESS)
        return result;

    /* Clean up old resources if swapchain is being recreated */
    destroy_staging(d);
    free(d->swapchain_images);
    d->swapchain_images = NULL;

    d->swapchain        = *pSwapchain;
    d->swapchain_format = pCreateInfo->imageFormat;
    d->swapchain_extent = pCreateInfo->imageExtent;

    /* Reset damage accumulator on swapchain recreation */
    d->pending_damage_count = DAMAGE_RECT_NOT_PRESENT;
    d->damage_overflow = 0;

    /* Get swapchain images */
    d->fpGetSwapchainImagesKHR(device, *pSwapchain,
                               &d->swapchain_image_count, NULL);
    d->swapchain_images = malloc(sizeof(VkImage) * d->swapchain_image_count);
    d->fpGetSwapchainImagesKHR(device, *pSwapchain,
                               &d->swapchain_image_count, d->swapchain_images);

    /* Create staging resources for the new swapchain dimensions */
    create_staging(d);

    /* Update shared memory header */
    if (d->shm_header) {
        d->shm_header->frame_width  = pCreateInfo->imageExtent.width;
        d->shm_header->frame_height = pCreateInfo->imageExtent.height;
        d->shm_header->frame_format = (uint32_t)pCreateInfo->imageFormat;
        d->shm_header->frame_stride =
            pCreateInfo->imageExtent.width * BYTES_PER_PIXEL;
        __atomic_or_fetch(&d->shm_header->flags,
                          FLAG_LAYER_READY, __ATOMIC_SEQ_CST);
    }

    fprintf(stderr, "[FC_LAYER] Swapchain created: %ux%u format=%u images=%u\n",
            pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
            (uint32_t)pCreateInfo->imageFormat, d->swapchain_image_count);

    return VK_SUCCESS;
}

/* ---- vkDestroySwapchainKHR ---- */

static VKAPI_ATTR void VKAPI_CALL
FC_DestroySwapchainKHR(VkDevice device,
                       VkSwapchainKHR swapchain,
                       const VkAllocationCallbacks *pAllocator)
{
    fc_device_data_t *d = find_device(device);
    if (!d) return;

    if (swapchain == d->swapchain) {
        destroy_staging(d);
        free(d->swapchain_images);
        d->swapchain_images      = NULL;
        d->swapchain_image_count = 0;
        d->swapchain             = VK_NULL_HANDLE;
    }

    d->fpDestroySwapchainKHR(device, swapchain, pAllocator);
}

/* ---- vkGetSwapchainImagesKHR ---- */

static VKAPI_ATTR VkResult VKAPI_CALL
FC_GetSwapchainImagesKHR(VkDevice device,
                         VkSwapchainKHR swapchain,
                         uint32_t *pSwapchainImageCount,
                         VkImage *pSwapchainImages)
{
    fc_device_data_t *d = find_device(device);
    if (!d) return VK_ERROR_DEVICE_LOST;
    return d->fpGetSwapchainImagesKHR(device, swapchain,
                                      pSwapchainImageCount, pSwapchainImages);
}

/* ---- vkQueuePresentKHR ---- */

static VKAPI_ATTR VkResult VKAPI_CALL
FC_QueuePresentKHR(VkQueue queue,
                   const VkPresentInfoKHR *pPresentInfo)
{
    fc_device_data_t *d = find_device_by_queue(queue);
    if (d && d->shm_header && d->staging_mapped &&
        pPresentInfo->swapchainCount > 0 &&
        d->swapchain_images)
    {
        accumulate_damage(d, pPresentInfo);
        uint32_t image_idx = pPresentInfo->pImageIndices[0];
        if (image_idx < d->swapchain_image_count) {
            capture_frame(d, queue, d->swapchain_images[image_idx]);
        }
    }

    return d ? d->fpQueuePresentKHR(queue, pPresentInfo)
             : VK_ERROR_DEVICE_LOST;
}

/* ================================================================== */
/* Dispatch (vkGet*ProcAddr)                                           */
/* ================================================================== */

#define INTERCEPT(fn) if (strcmp(pName, "vk" #fn) == 0) return (PFN_vkVoidFunction)FC_##fn

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
FC_GetDeviceProcAddr(VkDevice device, const char *pName) {
    /* Device-level intercepts */
    INTERCEPT(DestroyDevice);
    INTERCEPT(CreateSwapchainKHR);
    INTERCEPT(DestroySwapchainKHR);
    INTERCEPT(GetSwapchainImagesKHR);
    INTERCEPT(QueuePresentKHR);
    INTERCEPT(GetDeviceProcAddr);

    fc_device_data_t *d = find_device(device);
    if (d)
        return d->fpGetDeviceProcAddr(device, pName);
    return NULL;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
FC_GetInstanceProcAddr(VkInstance instance, const char *pName) {
    /* Instance-level intercepts */
    INTERCEPT(CreateInstance);
    INTERCEPT(DestroyInstance);
    INTERCEPT(CreateDevice);

    /* Device-level intercepts also need to be returned here because the
     * loader calls vkGetInstanceProcAddr for everything before devices
     * are created. */
    INTERCEPT(DestroyDevice);
    INTERCEPT(CreateSwapchainKHR);
    INTERCEPT(DestroySwapchainKHR);
    INTERCEPT(GetSwapchainImagesKHR);
    INTERCEPT(QueuePresentKHR);
    INTERCEPT(GetDeviceProcAddr);

    if (instance) {
        fc_instance_data_t *data = find_instance(instance);
        if (data)
            return data->fpGetInstanceProcAddr(instance, pName);
    }
    return NULL;
}

#undef INTERCEPT

/* ================================================================== */
/* Public symbols required by the Vulkan loader                        */
/*                                                                     */
/* The loader dlsym()s these even when interface v2 negotiation sets   */
/* the function pointers via the negotiate callback.                   */
/* ================================================================== */

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    return FC_GetInstanceProcAddr(instance, pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    return FC_GetDeviceProcAddr(device, pName);
}

/* ================================================================== */
/* Layer negotiation entry point                                       */
/* ================================================================== */

VKAPI_ATTR VkResult VKAPI_CALL
FC_NegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface *pVersionStruct)
{
    if (!pVersionStruct ||
        pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    /* We support interface version 2 (GetInstanceProcAddr +
     * GetDeviceProcAddr + GetPhysicalDeviceProcAddr) */
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr  = FC_GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr    = FC_GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    }

    return VK_SUCCESS;
}
