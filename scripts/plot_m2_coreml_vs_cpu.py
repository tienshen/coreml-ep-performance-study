#!/usr/bin/env python3
"""
Plot CPU-only vs CoreML+CPU results on Apple M2.

This script looks for per-model CSV files, for example:

  results/raw/m2_coreml_vs_cpu_bert-base-uncased.csv
  results/raw/m2_coreml_vs_cpu_tiny-systems-bert.csv

Each CSV is expected to have columns like:
  id, provider_mode, seq_len, batch_size,
  n_warmup, n_iters, model_name,
  latency_p50_ms, latency_p95_ms, throughput_samples_per_s

For each CSV, this script generates two plots:

  plots/m2_coreml_vs_cpu_<model>_latency.png
  plots/m2_coreml_vs_cpu_<model>_throughput.png

Usage (from repo root):

  python3 scripts/plot_m2_coreml_vs_cpu.py

Optional arguments:

  --results-dir   Directory containing CSVs (default: results/raw)
  --glob-pattern  Pattern of CSV filenames (default: m2_coreml_vs_cpu_*.csv)
  --batch-size    Batch size to filter on (default: 1)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_name_for_filename(name: str) -> str:
    """Make a safe string for use in filenames."""
    return (
        name.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


def plot_for_csv(csv_path: Path, out_dir: Path, batch_size: int) -> None:
    print(f"[INFO] Processing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic column checks
    required_cols = {
        "provider_mode",
        "seq_len",
        "batch_size",
        "latency_p50_ms",
        "throughput_samples_per_s",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Skipping {csv_path.name}, missing columns: {missing}")
        return

    # Filter by batch size
    df = df[df["batch_size"] == batch_size]
    if df.empty:
        print(f"[WARN] No rows with batch_size={batch_size} in {csv_path.name}, skipping.")
        return

    # Try to get model name from column; fall back to filename
    if "model_name" in df.columns and df["model_name"].notna().any():
        model_name = str(df["model_name"].iloc[0])
    else:
        # Strip prefix/suffix from filename like "m2_coreml_vs_cpu_<model>.csv"
        stem = csv_path.stem  # e.g. "m2_coreml_vs_cpu_bert-base-uncased"
        prefix = "m2_coreml_vs_cpu_"
        model_name = stem[len(prefix):] if stem.startswith(prefix) else stem

    safe_model = safe_name_for_filename(model_name)

    # Map provider_mode to nicer legend labels
    label_map = {
        "cpu_only": "CPU-only",
        "coreml_plus_cpu": "CoreML + CPU fallback",
    }
    if "provider_mode" in df.columns:
        df["mode_label"] = df["provider_mode"].map(label_map).fillna(df["provider_mode"])
    else:
        df["mode_label"] = "unknown"

    # Sort for nicer lines
    df = df.sort_values(["seq_len", "provider_mode"])

    # ---- Latency plot ----
    plt.figure()
    for mode, grp in df.groupby("mode_label"):
        plt.plot(
            grp["seq_len"],
            grp["latency_p50_ms"],
            marker="o",
            linestyle="-",
            label=mode,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("p50 latency (ms)")
    plt.title(
        f"Apple M2: p50 latency vs sequence length\n"
        f"(batch={batch_size}, model={model_name})"
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    latency_path = out_dir / f"m2_coreml_vs_cpu_{safe_model}_latency.png"
    plt.savefig(latency_path, bbox_inches="tight", dpi=150)

    # ---- Throughput plot ----
    plt.figure()
    for mode, grp in df.groupby("mode_label"):
        plt.plot(
            grp["seq_len"],
            grp["throughput_samples_per_s"],
            marker="o",
            linestyle="-",
            label=mode,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Throughput (inferences/sec)")
    plt.title(
        f"{mode_name} on Apple M2: throughput vs sequence length\n"
        f"(batch={batch_size}, model={model_name})"
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    throughput_path = out_dir / f"m2_coreml_vs_cpu_{safe_model}_throughput.png"
    plt.savefig(throughput_path, bbox_inches="tight", dpi=150)

    print(f"[INFO] Saved latency plot to    {latency_path}")
    print(f"[INFO] Saved throughput plot to {throughput_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/raw",
        help="Directory containing per-model CSV result files "
             "(default: results/raw).",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="m2_coreml_vs_cpu_*.csv",
        help="Glob pattern for CSV files (default: m2_coreml_vs_cpu_*.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to filter on for plotting (default: 1).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path("results/plots/m2_coreml_vs_cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(results_dir.glob(args.glob_pattern))
    if not csv_files:
        print(f"[WARN] No CSV files found in {results_dir} matching {args.glob_pattern}")
        return

    print(f"[INFO] Found {len(csv_files)} CSV file(s) to plot in {results_dir}")

    for csv_path in csv_files:
        try:
            plot_for_csv(csv_path, out_dir, batch_size=args.batch_size)
        except Exception as e:
            print(f"[ERROR] Failed to plot {csv_path.name}: {e}")


if __name__ == "__main__":
    main()
