# Eliminating the Encoder 1 GPU Copy via Fake Swapchain

## Current State

After the Vulkan-CUDA interop work, the frame paths are:

```
UNCHANGED  → classify_frame() → no GPU work
CHANGED    → vkCmdCopyImageToBuffer(full frame → enc1_buf) → NVENC
VIDEO_ONLY → vkCmdCopyImageToBuffer(region → enc2_buf)     → NVENC
```

The CHANGED path still requires a GPU-internal copy from the swapchain image
to our device-local buffer. This is fast (~0.1ms at 1280x720) but is the last
remaining copy in the encoder 1 pipeline.

## Why the Copy Exists

We can't export swapchain image memory to CUDA because swapchain images are
allocated by the WSI presentation engine, not by us. We can't add
`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` at allocation time.

## Key Insight

Chrome runs on Xdummy. Nobody is looking at the presented frames. We don't
need a real swapchain at all. The layer can give Chrome fake swapchain images
that we fully control, encode from them directly, and never present anything.

## Proposed Solution: Fake Swapchain

### Intercepted Functions

1. **`vkCreateSwapchainKHR`**: Don't create a real swapchain. Instead allocate
   N `VkImage`s with:
   - Same format, extent, and usage as requested (+ `TRANSFER_SRC`)
   - `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` on the memory
   - Export each image's fd at creation time
   - Import each fd into CUDA → get a dev_ptr per image
   - Return a fake `VkSwapchainKHR` handle

2. **`vkGetSwapchainImagesKHR`**: Return our VkImages.

3. **`vkAcquireNextImageKHR`**: Round-robin through our images. Signal the
   semaphore/fence immediately (no real acquire needed).

4. **`vkQueuePresentKHR`**: The presented image is ours. Classify, then:
   - UNCHANGED → no GPU work
   - CHANGED → NVENC encodes directly from the image's CUDA ptr (zero copy!)
   - VIDEO_ONLY → `vkCmdCopyImageToBuffer` region crop → NVENC
   - Return `VK_SUCCESS` without presenting anything

5. **`vkDestroySwapchainKHR`**: Free our VkImages and CUDA mappings.

### Encoder 2 (Region)

Still needs a copy — NVENC can't encode from an arbitrary sub-rect of a
VkImage. The `vkCmdCopyImageToBuffer` with `imageOffset` into a device-local
buffer remains the right approach for region encoding.

### Net Effect

```
UNCHANGED  → zero GPU work (same as now)
CHANGED    → NVENC from CUDA-mapped fake swapchain image (TRUE zero copy)
VIDEO_ONLY → vkCmdCopyImageToBuffer(region → enc2_buf) → NVENC (same as now)
```

## Complexity

Moderate — we already intercept all swapchain functions. Main new work:

- Allocate/manage N exportable VkImages instead of letting WSI do it
- Handle `vkAcquireNextImageKHR` (round-robin + immediate semaphore signal)
- CUDA import of VkImage memory (needs `cuExternalMemoryGetMappedMipmappedArray`
  or linear-tiling images with `cuExternalMemoryGetMappedBuffer`)
- Swapchain recreation (free old images, allocate new ones)

## Open Questions

- **Tiling**: Chrome renders into these images. With `VK_IMAGE_TILING_OPTIMAL`,
  CUDA may not be able to read the pixels linearly. We might need
  `VK_IMAGE_TILING_LINEAR`, which could affect Chrome's render performance.
  Alternatively, CUDA's `cuArray` path may handle optimal tiling natively.
- **Semaphore signaling**: `vkAcquireNextImageKHR` returns a semaphore that
  Chrome waits on before rendering. We need to signal it immediately or
  pre-signal it, which may require timeline semaphores or a dummy submit.
- **Multiple images in flight**: Chrome may acquire image N+1 before
  presenting image N. Need to track which images are in-flight and not
  re-acquire an image that Chrome is still rendering into.
- **Is ~0.1ms/frame worth it?** At 25fps the GPU copy is 0.25% of frame time.
  The real win is architectural simplicity — one fewer buffer, one fewer copy
  operation, and CHANGED frames have the same zero-copy property as the rest.
