/*
 * gpu_meter.c — GPU metrics wrapper for command execution
 *
 * Usage:
 *   gpu_meter [--period seconds] command [args...]
 *
 * Examples:
 *   gpu_meter python3 netflix_encoder.py              # Report at end
 *   gpu_meter --period 1 ffplay video.mkv            # Report every 1 second
 *
 * Monitors GPU utilization, memory, power, temperature while running a command.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <pthread.h>
#include <dlfcn.h>
#include <signal.h>
#include <time.h>

/* NVML types and constants */
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

#define NVML_SUCCESS 0
#define NVML_TEMPERATURE_GPU 0

/* NVML function pointers */
static nvmlReturn_t (*nvmlInit_v2)(void) = NULL;
static nvmlReturn_t (*nvmlShutdown)(void) = NULL;
static nvmlReturn_t (*nvmlDeviceGetCount_v2)(unsigned int *) = NULL;
static nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t *) = NULL;
static nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, unsigned int *, unsigned int *) = NULL;
static nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, void *) = NULL;
static nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int *) = NULL;
static nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int *) = NULL;
static nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int) = NULL;

/* Metrics snapshot */
typedef struct {
    unsigned int gpu_util;
    unsigned int mem_util;
    unsigned int power_mw;
    unsigned int temp_c;
    unsigned long mem_used_mb;
    unsigned long mem_total_mb;
    unsigned long timestamp_us;
} gpu_metrics_t;

typedef struct {
    volatile int running;
    pthread_t monitor_thread;
    unsigned int period_us;
    nvmlDevice_t device;
    gpu_metrics_t current;
    gpu_metrics_t peak;
    unsigned int sample_count;
} monitor_state_t;

/* Load NVML library and resolve symbols */
static int nvml_load(void) {
    void *handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error: Could not load libnvidia-ml.so.1\n");
        fprintf(stderr, "  Install NVIDIA driver (includes NVML)\n");
        return 0;
    }

    nvmlInit_v2 = dlsym(handle, "nvmlInit_v2");
    nvmlShutdown = dlsym(handle, "nvmlShutdown");
    nvmlDeviceGetCount_v2 = dlsym(handle, "nvmlDeviceGetCount_v2");
    nvmlDeviceGetHandleByIndex_v2 = dlsym(handle, "nvmlDeviceGetHandleByIndex_v2");
    nvmlDeviceGetUtilizationRates = dlsym(handle, "nvmlDeviceGetUtilizationRates");
    nvmlDeviceGetMemoryInfo = dlsym(handle, "nvmlDeviceGetMemoryInfo");
    nvmlDeviceGetPowerUsage = dlsym(handle, "nvmlDeviceGetPowerUsage");
    nvmlDeviceGetTemperature = dlsym(handle, "nvmlDeviceGetTemperature");
    nvmlDeviceGetName = dlsym(handle, "nvmlDeviceGetName");

    if (!nvmlInit_v2 || !nvmlDeviceGetCount_v2) {
        fprintf(stderr, "Error: Could not resolve NVML symbols\n");
        dlclose(handle);
        return 0;
    }

    return 1;
}

/* Get current GPU metrics */
static int get_metrics(nvmlDevice_t device, gpu_metrics_t *metrics) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    metrics->timestamp_us = ts.tv_sec * 1000000UL + ts.tv_nsec / 1000;

    if (nvmlDeviceGetUtilizationRates(device, &metrics->gpu_util, &metrics->mem_util) != NVML_SUCCESS) {
        metrics->gpu_util = 0;
        metrics->mem_util = 0;
    }

    if (nvmlDeviceGetPowerUsage(device, &metrics->power_mw) != NVML_SUCCESS) {
        metrics->power_mw = 0;
    }

    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &metrics->temp_c) != NVML_SUCCESS) {
        metrics->temp_c = 0;
    }

    /* Memory info structure: uint64_t total, uint64_t free, uint64_t used */
    unsigned long mem_info[3] = {0};
    if (nvmlDeviceGetMemoryInfo(device, &mem_info) != NVML_SUCCESS) {
        metrics->mem_total_mb = 0;
        metrics->mem_used_mb = 0;
    } else {
        metrics->mem_total_mb = mem_info[0] / 1024 / 1024;
        metrics->mem_used_mb = mem_info[2] / 1024 / 1024;
    }

    return 1;
}

/* Monitoring thread */
static void* monitor_thread_func(void *arg) {
    monitor_state_t *state = (monitor_state_t *)arg;
    struct timespec sleep_time;

    while (state->running) {
        get_metrics(state->device, &state->current);
        state->sample_count++;

        /* Update peak values */
        if (state->current.gpu_util > state->peak.gpu_util)
            state->peak.gpu_util = state->current.gpu_util;
        if (state->current.mem_util > state->peak.mem_util)
            state->peak.mem_util = state->current.mem_util;
        if (state->current.power_mw > state->peak.power_mw)
            state->peak.power_mw = state->current.power_mw;
        if (state->current.temp_c > state->peak.temp_c)
            state->peak.temp_c = state->current.temp_c;

        if (state->period_us > 0) {
            /* Print current metrics */
            printf("[%.2fs] GPU: %3u%% | Mem: %3u%% (%lu/%lu MB) | Power: %u W | Temp: %u°C\n",
                   (double)state->current.timestamp_us / 1000000.0,
                   state->current.gpu_util,
                   state->current.mem_util,
                   state->current.mem_used_mb,
                   state->current.mem_total_mb,
                   state->current.power_mw / 1000,
                   state->current.temp_c);
            fflush(stdout);

            sleep_time.tv_sec = state->period_us / 1000000;
            sleep_time.tv_nsec = (state->period_us % 1000000) * 1000;
            nanosleep(&sleep_time, NULL);
        } else {
            /* Just sleep a bit to avoid busy loop */
            sleep_time.tv_sec = 0;
            sleep_time.tv_nsec = 100000000;  /* 100ms */
            nanosleep(&sleep_time, NULL);
        }
    }

    return NULL;
}

/* Print metrics summary */
static void print_summary(monitor_state_t *state, gpu_metrics_t *baseline, double elapsed_sec) {
    printf("\n");
    printf("================================================\n");
    printf("GPU Metrics Summary\n");
    printf("================================================\n");
    printf("  Elapsed Time:      %.2f seconds\n", elapsed_sec);
    printf("  Samples:           %u\n", state->sample_count);
    printf("\n");
    printf("  GPU Utilization:   %3u%% (peak: %u%%, baseline: %u%%)\n",
           state->current.gpu_util, state->peak.gpu_util, baseline->gpu_util);
    printf("  Memory:            %3u%% (peak: %u%%, baseline: %u%%)\n",
           state->current.mem_util, state->peak.mem_util, baseline->mem_util);
    printf("                     %lu / %lu MB used (baseline: %lu MB)\n",
           state->current.mem_used_mb, state->current.mem_total_mb, baseline->mem_used_mb);
    printf("  Power:             %u W (peak: %u W)\n",
           state->current.power_mw / 1000, state->peak.power_mw / 1000);
    printf("  Temperature:       %u°C (peak: %u°C, baseline: %u°C)\n",
           state->current.temp_c, state->peak.temp_c, baseline->temp_c);
    printf("================================================\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--period seconds] command [args...]\n", argv[0]);
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  %s python3 netflix_encoder.py\n", argv[0]);
        fprintf(stderr, "  %s --period 1 ffplay video.mkv\n", argv[0]);
        return 1;
    }

    /* Parse arguments */
    unsigned int period_sec = 0;
    int cmd_start = 1;

    if (strcmp(argv[1], "--period") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: --period requires seconds and command\n");
            return 1;
        }
        period_sec = atoi(argv[2]);
        cmd_start = 3;
    }

    /* Load NVML */
    if (!nvml_load()) {
        return 1;
    }

    /* Initialize NVML */
    if (nvmlInit_v2() != NVML_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize NVML\n");
        return 1;
    }

    /* Get device */
    unsigned int device_count;
    if (nvmlDeviceGetCount_v2(&device_count) != NVML_SUCCESS || device_count == 0) {
        fprintf(stderr, "Error: No NVIDIA GPU found\n");
        nvmlShutdown();
        return 1;
    }

    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex_v2(0, &device) != NVML_SUCCESS) {
        fprintf(stderr, "Error: Could not get GPU device handle\n");
        nvmlShutdown();
        return 1;
    }

    /* Print device info */
    char device_name[256];
    nvmlDeviceGetName(device, device_name, sizeof(device_name));
    printf("GPU: %s\n", device_name);

    /* Get baseline reading */
    gpu_metrics_t baseline;
    get_metrics(device, &baseline);
    printf("Baseline: GPU %3u%% | Mem %3u%% (%lu/%lu MB) | Temp %u°C\n",
           baseline.gpu_util, baseline.mem_util,
           baseline.mem_used_mb, baseline.mem_total_mb,
           baseline.temp_c);

    if (period_sec > 0) {
        printf("Reporting every %u seconds...\n\n", period_sec);
    } else {
        printf("\n");
    }

    /* Start monitoring */
    monitor_state_t state = {
        .running = 1,
        .period_us = period_sec * 1000000,
        .device = device,
        .sample_count = 0,
    };

    memset(&state.peak, 0, sizeof(state.peak));
    pthread_create(&state.monitor_thread, NULL, monitor_thread_func, &state);

    /* Fork and execute command */
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        state.running = 0;
        nvmlShutdown();
        return 1;
    }

    if (pid == 0) {
        /* Child: execute command */
        execvp(argv[cmd_start], &argv[cmd_start]);
        perror("execvp");
        exit(127);
    }

    /* Parent: wait for child */
    int status;
    waitpid(pid, &status, 0);
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    /* Stop monitoring */
    state.running = 0;
    pthread_join(state.monitor_thread, NULL);

    double elapsed_sec = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    /* Print summary */
    print_summary(&state, &baseline, elapsed_sec);

    /* Cleanup */
    nvmlShutdown();

    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
