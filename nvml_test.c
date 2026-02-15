#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

#define NVML_SUCCESS 0

int main() {
    printf("Testing NVML encoder/decoder functions...\n\n");

    void *handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!handle) {
        printf("Error: Could not load libnvidia-ml.so.1\n");
        return 1;
    }

    /* Try different function name variations */
    const char *func_names[] = {
        "nvmlDeviceGetEncoderUtilization",
        "nvmlDeviceGetDecoderUtilization",
        "nvmlDeviceGetEncoderStats",
        "nvmlDeviceGetDecoderStats",
        "nvmlDeviceGetEncoderCapacity",
        "nvmlDeviceGetEncoderProcesses",
        NULL
    };

    printf("Available NVML encoder/decoder functions:\n");
    for (int i = 0; func_names[i]; i++) {
        void *sym = dlsym(handle, func_names[i]);
        if (sym) {
            printf("  ✓ %s\n", func_names[i]);
        } else {
            printf("  ✗ %s\n", func_names[i]);
        }
    }

    printf("\nTrying to call available functions...\n\n");

    /* Initialize NVML */
    nvmlReturn_t (*nvmlInit_v2)(void) = dlsym(handle, "nvmlInit_v2");
    nvmlReturn_t (*nvmlShutdown)(void) = dlsym(handle, "nvmlShutdown");
    nvmlReturn_t (*nvmlDeviceGetCount_v2)(unsigned int *) = dlsym(handle, "nvmlDeviceGetCount_v2");
    nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t *) = dlsym(handle, "nvmlDeviceGetHandleByIndex_v2");

    if (!nvmlInit_v2 || !nvmlDeviceGetCount_v2) {
        printf("Error: Core NVML functions not found\n");
        dlclose(handle);
        return 1;
    }

    if (nvmlInit_v2() != NVML_SUCCESS) {
        printf("Error: Failed to initialize NVML\n");
        dlclose(handle);
        return 1;
    }

    unsigned int device_count;
    nvmlDeviceGetCount_v2(&device_count);

    if (device_count == 0) {
        printf("No GPU found\n");
        nvmlShutdown();
        dlclose(handle);
        return 1;
    }

    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex_v2(0, &device);

    printf("Testing encoder utilization function signatures...\n\n");

    /* Test the actual encoder function with different signatures */
    typedef nvmlReturn_t (*enc_util_2params)(nvmlDevice_t, unsigned int *);
    typedef nvmlReturn_t (*enc_util_3params)(nvmlDevice_t, unsigned int *, unsigned int *);

    enc_util_3params nvmlGetEnc3 = dlsym(handle, "nvmlDeviceGetEncoderUtilization");
    if (nvmlGetEnc3) {
        unsigned int util = 0, samplingPeriod = 0;
        nvmlReturn_t ret = nvmlGetEnc3(device, &util, &samplingPeriod);
        printf("nvmlDeviceGetEncoderUtilization(device, &util, &period):\n");
        printf("  Return: %d (0=success)\n", ret);
        printf("  Encoder util: %u%%\n", util);
        printf("  Sampling period: %u ms\n\n", samplingPeriod);
    }

    enc_util_2params nvmlGetEnc2 = dlsym(handle, "nvmlDeviceGetEncoderUtilization");
    if (nvmlGetEnc2) {
        unsigned int util = 0;
        nvmlReturn_t ret = nvmlGetEnc2(device, &util);
        printf("nvmlDeviceGetEncoderUtilization(device, &util):\n");
        printf("  Return: %d\n", ret);
        printf("  Encoder util: %u%%\n\n", util);
    }

    /* Try decoder too */
    enc_util_3params nvmlGetDec3 = dlsym(handle, "nvmlDeviceGetDecoderUtilization");
    if (nvmlGetDec3) {
        unsigned int util = 0, samplingPeriod = 0;
        nvmlReturn_t ret = nvmlGetDec3(device, &util, &samplingPeriod);
        printf("nvmlDeviceGetDecoderUtilization(device, &util, &period):\n");
        printf("  Return: %d\n", ret);
        printf("  Decoder util: %u%%\n", util);
        printf("  Sampling period: %u ms\n\n", samplingPeriod);
    }

    nvmlShutdown();
    dlclose(handle);

    printf("Test complete.\n");
    return 0;
}
