/*
 * frame_capture_layer.h -- Internal types for VK_LAYER_CHROME_frame_capture.
 */

#ifndef FRAME_CAPTURE_LAYER_H
#define FRAME_CAPTURE_LAYER_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include "shm_protocol.h"
#include "nvenc_encode.h"

/* We only ever expect one instance and one device from Chrome. */
#define MAX_INSTANCES 4
#define MAX_DEVICES   4

/* Helper: the first sizeof(void*) of any dispatchable handle is the
 * loader dispatch table pointer -- used as our lookup key. */
#define GET_DISPATCH_KEY(obj) (*(void **)(obj))

/* ------------------------------------------------------------------ */
/* Per-instance data                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    void                          *dispatch_key;
    VkInstance                     instance;
    PFN_vkGetInstanceProcAddr      fpGetInstanceProcAddr;
    PFN_vkDestroyInstance          fpDestroyInstance;
    PFN_vkEnumerateDeviceExtensionProperties fpEnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties  fpGetPhysicalDeviceMemoryProperties;
} fc_instance_data_t;

/* ------------------------------------------------------------------ */
/* Per-device data                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    void                          *dispatch_key;
    VkDevice                       device;
    VkPhysicalDevice               physical_device;

    /* Next-layer device functions we need to call through */
    PFN_vkGetDeviceProcAddr        fpGetDeviceProcAddr;
    PFN_vkDestroyDevice            fpDestroyDevice;

    /* Swapchain */
    PFN_vkCreateSwapchainKHR       fpCreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR      fpDestroySwapchainKHR;
    PFN_vkGetSwapchainImagesKHR    fpGetSwapchainImagesKHR;

    /* Queue / present */
    PFN_vkQueuePresentKHR          fpQueuePresentKHR;
    PFN_vkQueueSubmit              fpQueueSubmit;
    PFN_vkQueueWaitIdle            fpQueueWaitIdle;

    /* Command buffers */
    PFN_vkCreateCommandPool        fpCreateCommandPool;
    PFN_vkDestroyCommandPool       fpDestroyCommandPool;
    PFN_vkAllocateCommandBuffers   fpAllocateCommandBuffers;
    PFN_vkFreeCommandBuffers       fpFreeCommandBuffers;
    PFN_vkBeginCommandBuffer       fpBeginCommandBuffer;
    PFN_vkEndCommandBuffer         fpEndCommandBuffer;
    PFN_vkResetCommandBuffer       fpResetCommandBuffer;

    /* Transfer commands */
    PFN_vkCmdPipelineBarrier       fpCmdPipelineBarrier;
    PFN_vkCmdCopyImageToBuffer     fpCmdCopyImageToBuffer;

    /* Memory / buffer */
    PFN_vkCreateBuffer             fpCreateBuffer;
    PFN_vkDestroyBuffer            fpDestroyBuffer;
    PFN_vkGetBufferMemoryRequirements fpGetBufferMemoryRequirements;
    PFN_vkAllocateMemory           fpAllocateMemory;
    PFN_vkFreeMemory               fpFreeMemory;
    PFN_vkBindBufferMemory         fpBindBufferMemory;
    PFN_vkMapMemory                fpMapMemory;
    PFN_vkUnmapMemory              fpUnmapMemory;

    /* Fence */
    PFN_vkCreateFence              fpCreateFence;
    PFN_vkDestroyFence             fpDestroyFence;
    PFN_vkWaitForFences            fpWaitForFences;
    PFN_vkResetFences              fpResetFences;

    /* Physical device memory properties (for finding host-visible heap) */
    PFN_vkGetPhysicalDeviceMemoryProperties fpGetPhysicalDeviceMemoryProperties;

    /* External memory (Vulkan-CUDA interop) */
    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;

    /* ---- Command resources (created when swapchain is created) ---- */
    VkCommandPool    cmd_pool;
    VkCommandBuffer  cmd_buf;
    VkFence          fence;
    uint32_t         queue_family;     /* queue family used for transfer */

    /* ---- Host staging buffer (debug pixel copy only) ---- */
    VkBuffer         staging_buf;
    VkDeviceMemory   staging_mem;
    void            *staging_mapped;   /* persistently mapped */
    VkDeviceSize     staging_size;

    /* ---- Encoder 1 GPU buffer (device-local, full frame, lazy-init) ---- */
    VkBuffer         enc1_buf;
    VkDeviceMemory   enc1_mem;
    VkDeviceSize     enc1_alloc_size;

    /* ---- Encoder 2 GPU buffer (device-local, region, recreated on size change) ---- */
    VkBuffer         enc2_buf;
    VkDeviceMemory   enc2_mem;
    VkDeviceSize     enc2_alloc_size;

    /* ---- Swapchain tracking ---- */
    VkSwapchainKHR   swapchain;
    VkImage         *swapchain_images;
    uint32_t         swapchain_image_count;
    VkFormat         swapchain_format;
    VkExtent2D       swapchain_extent;

    /* ---- Shared memory ---- */
    int              shm_fd;
    shm_header_t    *shm_header;
    uint8_t         *shm_data;         /* pointer past header to pixel region */

    /* ---- Damage rect accumulation across skipped frames ---- */
    shm_damage_rect_t pending_damage[MAX_DAMAGE_RECTS];
    uint32_t           pending_damage_count;  /* init to DAMAGE_RECT_NOT_PRESENT */
    int                damage_overflow;        /* set if > MAX rects accumulated */

    /* ---- Layer-side NVENC encoders (lazy-init) ---- */
    nvenc_ctx_t       *nvenc_enc1;     /* full-frame encoder */
    nvenc_ctx_t       *nvenc_enc2;     /* region encoder */
    int                enc2_width;     /* current encoder2 dimensions */
    int                enc2_height;
    int                enc2_needs_idr; /* force IDR on next encoder2 use */
} fc_device_data_t;

#endif /* FRAME_CAPTURE_LAYER_H */
