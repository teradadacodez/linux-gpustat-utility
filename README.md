# GPU Live Monitor

A lightweight Python utility for monitoring an NVIDIA GPU on Linux. It samples GPU utilization, VRAM usage, temperature, and power draw, displays a live Matplotlib dashboard, and saves every sample to a timestamped CSV file.

## Who should use it?

This project is useful for:

- **Machine-learning and deep-learning learners** who want to see whether PyTorch or another CUDA workload is using the GPU.
- **Students and developers** profiling GPU activity while training models or running inference.
- **Linux users** whose desktop System Monitor does not show NVIDIA GPU metrics.
- **Anyone investigating GPU memory usage, temperature, or power draw** during a workload.

It is intended as a small monitoring and logging tool, not a replacement for professional performance profilers.

## Features

- Live graphs for GPU utilization, VRAM usage, temperature, and power draw.
- CSV logging for later analysis in Pandas, LibreOffice Calc, or spreadsheet software.
- Configurable sampling interval and graph window.
- CSV-only mode for headless use.
- Handles unavailable metrics (for example, when a driver does not expose power draw).

## Requirements

- Linux with a working NVIDIA driver.
- `nvidia-smi` available on your `PATH`.
- Python 3.9 or newer.
- Matplotlib.

The tool uses NVIDIA's `nvidia-smi` command-line utility. It does not install or configure GPU drivers, CUDA, or PyTorch for you. Other GPU vendors are not currently supported.

## Installation

### 1. Check your NVIDIA driver

Run:

```bash
nvidia-smi
```

If this command is missing or reports a driver error, resolve the NVIDIA driver setup before using this tool.

### 2. Get the project

Clone the repository after publishing it:

```bash
git clone https://github.com/YOUR-USERNAME/gpu-live-monitor.git
cd gpu-live-monitor
```

Or download the repository ZIP from GitHub and extract it.

### 3. Install the Python dependency

Using pip:

```bash
python -m pip install -r requirements.txt
```

On an Anaconda setup, activate the environment where you want to run the tool first. For example:

```bash
conda activate base
python -m pip install -r requirements.txt
```

If Matplotlib is already installed in that environment, no additional Python package installation is needed.

## Usage

### Start live monitoring

```bash
python gpu_monitor.py
```

The program opens a window with four graphs and creates a CSV file in `~/gpu_logs/`.

### Change the sampling interval

Sample every two seconds:

```bash
python gpu_monitor.py --interval 2
```

A shorter interval gives more frequent readings but invokes `nvidia-smi` more often.

### Show a longer or shorter graph history

Show the most recent 300 samples:

```bash
python gpu_monitor.py --window 300
```

The CSV continues to record the whole session; `--window` only changes how many recent samples are shown in the graph.

### Choose where logs are saved

```bash
python gpu_monitor.py --output-dir ./logs
```

### Log without a graph

Useful on a remote/headless session:

```bash
python gpu_monitor.py --no-plot
```

Stop the program with `Ctrl+C`. When the graph is enabled, closing its window also stops the monitor.

## CSV output

By default, logs are written to:

```text
~/gpu_logs/gpu_YYYYMMDD_HHMMSS.csv
```

Each row contains:

| Column | Meaning |
|---|---|
| `timestamp` | Local time when the sample was recorded |
| `gpu_utilization_percent` | GPU utilization percentage |
| `memory_used_mib` | GPU memory currently used, in MiB |
| `memory_total_mib` | Total GPU memory, in MiB |
| `temperature_celsius` | GPU temperature in °C |
| `power_draw_watts` | GPU power draw in watts |

A metric can be blank if the NVIDIA driver does not expose it on your hardware. The CSV is flushed after each sample so data is written incrementally.

### Plot the log later with Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("gpu_logs/your-log-file.csv", parse_dates=["timestamp"])

df.plot(x="timestamp", y="gpu_utilization_percent", legend=False)
plt.ylabel("GPU utilization (%)")
plt.tight_layout()
plt.show()
```

Replace `your-log-file.csv` with the actual CSV filename.

## Troubleshooting

### `nvidia-smi was not found`

Check whether `nvidia-smi` works in the same terminal. This program requires a working NVIDIA driver and the NVIDIA utility on `PATH`.

### Power or temperature shows blank

Not all NVIDIA GPU/driver combinations expose every metric. Unsupported readings are left blank in the CSV and omitted from the plotted line.

### The graph window does not open

Make sure you're running in a graphical desktop session and that Matplotlib is installed:

```bash
python -m pip install matplotlib
```

For a headless session, use `--no-plot`.

### GPU utilization stays near 0%

The monitor only reports activity; it does not create GPU work. Start a CUDA workload (such as model training or inference) to observe usage.

## Limitations

- NVIDIA GPUs and `nvidia-smi` only; AMD and Intel GPUs are not supported.
- The graph displays the most recent samples, while the CSV retains the complete run.
- GPU utilization is a sampled metric and may miss very short bursts of activity.
- Some metrics, especially power draw, may not be available on every device.
- On systems with multiple NVIDIA GPUs, this version monitors the first GPU reported by `nvidia-smi`.

## Contributing

Issues and pull requests are welcome. Please include your Linux distribution, Python version, NVIDIA driver version, and relevant `nvidia-smi` output when reporting a problem. Redact machine-specific or sensitive information before posting logs publicly.

## License

Choose and add a license before publishing this project publicly. Without a license, default copyright restrictions apply. MIT is a common permissive option for small utilities, but choose the license that matches your intentions.
