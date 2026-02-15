# gpu_meter — GPU Metrics Wrapper

Monitor GPU utilization, memory, power, and temperature while running any command.

## Building

```bash
gcc -o gpu_meter gpu_meter.c -pthread -ldl
```

Requires:
- NVIDIA driver (includes NVML library `libnvidia-ml.so.1`)
- POSIX threads (`pthread`)
- Dynamic linker (`dl`)

## Usage

### Report at End (Default)

```bash
./gpu_meter command [args...]
```

Example:
```bash
./gpu_meter python3 netflix_encoder.py
```

Output:
```
GPU: NVIDIA GeForce RTX 3050

================================================
GPU Metrics Summary
================================================
  Elapsed Time:      47.20 seconds
  Samples:           236

  GPU Utilization:    87% (peak: 95%)
  Memory:             45% (peak: 62%)
                     3682 / 8192 MB used
  Power:             85 W (peak: 120 W)
  Temperature:       68°C (peak: 72°C)
================================================
```

### Periodic Reporting

```bash
./gpu_meter --period seconds command [args...]
```

Example:
```bash
./gpu_meter --period 1 ffplay video.mkv
```

Output (every 1 second):
```
GPU: NVIDIA GeForce RTX 3050
Reporting every 1 seconds...

[0.00s] GPU:  42% | Mem:  28% (2293/8192 MB) | Power: 75 W | Temp: 52°C
[1.00s] GPU:  78% | Mem:  45% (3682/8192 MB) | Power: 98 W | Temp: 58°C
[2.00s] GPU:  85% | Mem:  52% (4262/8192 MB) | Power: 112 W | Temp: 64°C

================================================
GPU Metrics Summary
================================================
  Elapsed Time:      3.00 seconds
  Samples:           3

  GPU Utilization:    85% (peak: 85%)
  Memory:             52% (peak: 52%)
                     4262 / 8192 MB used
  Power:             112 W (peak: 112 W)
  Temperature:       64°C (peak: 72°C)
================================================
```

## Metrics Tracked

- **GPU Utilization**: 0-100% of GPU compute engines
- **Memory Utilization**: 0-100% of GPU VRAM
- **Memory Used/Total**: In MB
- **Power**: In watts (0 if not available on device)
- **Temperature**: GPU die temperature in °C

Both current and peak values are reported.

## Implementation

Uses NVML (NVIDIA Management Library) via `libnvidia-ml.so.1`:
- Runtime dynamic linking (no compile-time NVIDIA SDK needed)
- Spawns monitoring thread while command runs
- Samples GPU metrics every 100ms (or custom period)
- Reports summary and optionally periodic snapshots

## Error Handling

- If `libnvidia-ml.so.1` not found: "Could not load libnvidia-ml.so.1"
- If no GPU found: "No NVIDIA GPU found"
- Command exit code is preserved
