#!/usr/bin/env python3
"""Log NVIDIA GPU metrics to CSV and display a live Matplotlib dashboard."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Missing dependency: matplotlib. Install it with:\n"
        "  python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)


QUERY_FIELDS = (
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)
CSV_FIELDS = (
    "timestamp",
    "gpu_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_celsius",
    "power_draw_watts",
)


def parse_metric(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_gpu_metrics() -> dict[str, Optional[float]]:
    query = ",".join(QUERY_FIELDS)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi was not found. Install/configure the NVIDIA driver "
            "and ensure nvidia-smi is on your PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("nvidia-smi did not respond within 5 seconds.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "No additional details from nvidia-smi."
        raise RuntimeError(f"nvidia-smi failed: {detail}") from exc

    # This simple monitor targets one GPU. If multiple GPUs are present,
    # nvidia-smi may return several lines; use the first GPU by default.
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    values = [part.strip() for part in line.split(",")]
    if len(values) != len(QUERY_FIELDS):
        raise RuntimeError(
            "Unexpected output from nvidia-smi. Check that the NVIDIA driver "
            "supports the requested query fields."
        )

    parsed = [parse_metric(value) for value in values]
    return dict(zip(QUERY_FIELDS, parsed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Log NVIDIA GPU utilization, VRAM, temperature, and power to CSV "
            "while displaying a live graph."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0). Must be positive.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Number of recent samples shown in the graph (default: 120).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "gpu_logs",
        help="Directory for CSV logs (default: ~/gpu_logs).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Log metrics to CSV without opening a graph.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval <= 0:
        print("Error: --interval must be greater than zero.", file=sys.stderr)
        return 2
    if args.window < 2:
        print("Error: --window must be at least 2.", file=sys.stderr)
        return 2
    if shutil.which("nvidia-smi") is None:
        print(
            "Error: nvidia-smi was not found. This tool requires a working "
            "NVIDIA driver and nvidia-smi.",
            file=sys.stderr,
        )
        return 1

    try:
        initial = read_gpu_metrics()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir.expanduser() / (
        f"gpu_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    sample_times: deque[datetime] = deque(maxlen=args.window)
    gpu_util: deque[Optional[float]] = deque(maxlen=args.window)
    memory_used: deque[Optional[float]] = deque(maxlen=args.window)
    temperatures: deque[Optional[float]] = deque(maxlen=args.window)
    power_draw: deque[Optional[float]] = deque(maxlen=args.window)

    fig = None
    axes = []
    lines = []
    if not args.no_plot:
        plt.ion()
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        fig.suptitle("Live NVIDIA GPU Monitor")
        series = [
            ("GPU utilization", "%", "tab:blue", gpu_util),
            ("VRAM usage", "MiB", "tab:orange", memory_used),
            ("GPU temperature", "°C", "tab:red", temperatures),
            ("GPU power draw", "W", "tab:green", power_draw),
        ]
        for ax, (title, ylabel, color, _) in zip(axes, series):
            line, = ax.plot([], [], color=color, linewidth=1.6)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            lines.append(line)
        axes[0].set_ylim(0, 100)
        if initial["memory.total"] is not None:
            axes[1].set_ylim(0, initial["memory.total"] * 1.05)
        axes[2].set_ylim(bottom=0)
        axes[3].set_ylim(bottom=0)
        axes[-1].set_xlabel("Recent samples (one sample per interval)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    print(f"GPU logging started. CSV: {log_path}")
    print(f"Sampling interval: {args.interval:g} second(s)")
    if args.no_plot:
        print("Graph disabled. Press Ctrl+C to stop.")
    else:
        print("Close the graph window or press Ctrl+C to stop.")

    try:
        with log_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            csv_file.flush()

            while args.no_plot or (fig is not None and plt.fignum_exists(fig.number)):
                started = time.monotonic()
                try:
                    metrics = read_gpu_metrics()
                except RuntimeError as exc:
                    print(f"Warning: {exc}", file=sys.stderr)
                    time.sleep(args.interval)
                    continue

                now = datetime.now()
                sample_times.append(now)
                gpu_util.append(metrics["utilization.gpu"])
                memory_used.append(metrics["memory.used"])
                temperatures.append(metrics["temperature.gpu"])
                power_draw.append(metrics["power.draw"])

                writer.writerow(
                    {
                        "timestamp": now.isoformat(timespec="seconds"),
                        "gpu_utilization_percent": metrics["utilization.gpu"],
                        "memory_used_mib": metrics["memory.used"],
                        "memory_total_mib": metrics["memory.total"],
                        "temperature_celsius": metrics["temperature.gpu"],
                        "power_draw_watts": metrics["power.draw"],
                    }
                )
                csv_file.flush()

                if not args.no_plot and fig is not None:
                    plotted_series = [gpu_util, memory_used, temperatures, power_draw]
                    for line, data in zip(lines, plotted_series):
                        y_values = [
                            float(value) if value is not None else float("nan")
                            for value in data
                        ]
                        line.set_data(range(len(y_values)), y_values)
                    for ax in axes:
                        ax.set_xlim(0, max(args.window - 1, len(sample_times) - 1))
                    if metrics["memory.total"] is not None:
                        axes[1].set_ylim(0, metrics["memory.total"] * 1.05)
                    fig.canvas.draw_idle()
                    plt.pause(0.01)

                elapsed = time.monotonic() - started
                time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        if fig is not None:
            plt.close(fig)

    print(f"Log saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
