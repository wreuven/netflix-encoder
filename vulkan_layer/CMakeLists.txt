cmake_minimum_required(VERSION 3.16)
project(chrome_frame_capture_layer C)

set(CMAKE_C_STANDARD 11)

find_package(Vulkan REQUIRED)

add_library(VkLayer_chrome_frame_capture SHARED
    frame_capture_layer.c
    nvenc_encode.c
)

target_include_directories(VkLayer_chrome_frame_capture PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/ffnvcodec
)

target_link_libraries(VkLayer_chrome_frame_capture PRIVATE
    rt
    pthread
    dl
)

set_target_properties(VkLayer_chrome_frame_capture PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "VkLayer_chrome_frame_capture"
)

# Copy manifest next to the .so
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/VK_LAYER_CHROME_frame_capture.json
    ${CMAKE_CURRENT_BINARY_DIR}/VK_LAYER_CHROME_frame_capture.json
    COPYONLY
)
