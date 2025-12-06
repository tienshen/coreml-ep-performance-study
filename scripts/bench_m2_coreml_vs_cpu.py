#!/usr/bin/env python3
"""
Benchmark CPU-only vs CoreML+CPU on Apple M2 for a given ONNX model.

Example:

  # BERT base
  python scripts/bench_m2_coreml_vs_cpu.py \
    --model-name bert-base-uncased

  # tiny random BERT
  python scripts/bench_m2_coreml_vs_cpu.py \
    --model-name hf-internal-testing/tiny-random-bert

By default, it looks for an ONNX file at:
  models/<model-name-with-slashes-replaced>.onnx

You can override that with --onnx-path.
"""

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass
class M2ExperimentConfig:
    id: str
    provider_mode: Literal["cpu_only", "coreml_plus_cpu"]
    seq_len: int
    batch_size: int
    n_warmup: int = 5
    n_iters: int = 50


def make_dummy_inputs(tokenizer, seq_len: int, batch_size: int) -> Dict[str, Any]:
    text = ["hello world"] * batch_size
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {k: v for k, v in enc.items()}


def build_session(onnx_path: Path, provider_mode: str, verbose_ort: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    # 2 = warnings+errors only, 1 = info+warnings+errors
    so.log_severity_level = 1 if verbose_ort else 2

    if provider_mode == "cpu_only":
        providers = ["CPUExecutionProvider"]
    elif provider_mode == "coreml_plus_cpu":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown provider_mode: {provider_mode}")

    sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=so, providers=providers)
    print(f"[INFO] Requested providers: {providers}")
    print(f"[INFO] Effective providers: {sess.get_providers()}")
    return sess


def run_ort_benchmark(
    cfg: M2ExperimentConfig,
    onnx_path: Path,
    tokenizer,
    model_name: str,
    verbose_ort: bool,
) -> Dict[str, Any]:
    print(f"\n=== Running {cfg.id} ({model_name}) ===")
    print(
        f"mode={cfg.provider_mode}, seq_len={cfg.seq_len}, "
        f"batch_size={cfg.batch_size}, iters={cfg.n_iters}"
    )

    sess = build_session(onnx_path, cfg.provider_mode, verbose_ort)
    inputs = make_dummy_inputs(tokenizer, cfg.seq_len, cfg.batch_size)

    # Warmup
    for _ in range(cfg.n_warmup):
        _ = sess.run(None, inputs)

    # Timed runs
    times = []
    for _ in range(cfg.n_iters):
        t0 = time.perf_counter()
        _ = sess.run(None, inputs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    latency_p50_ms = np.percentile(times, 50) * 1000
    latency_p95_ms = np.percentile(times, 95) * 1000
    throughput = (cfg.batch_size * cfg.n_iters) / times.sum()

    metrics = {
        "model_name": model_name,
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
        "throughput_samples_per_s": throughput,
    }
    print(
        f"[RESULT] p50={latency_p50_ms:.2f} ms, p95={latency_p95_ms:.2f} ms, "
        f"throughput={throughput:.2f} samples/s"
    )

    return {**asdict(cfg), **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model id, e.g. bert-base-uncased or hf-internal-testing/tiny-random-bert",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to ONNX model file. "
             "Default: models/<model-name-with-slashes-replaced>.onnx",
    )
    
    parser.add_argument(
        "--verbose-ort",
        action="store_true",
        help="Enable more verbose ONNX Runtime logs (info level).",
    )
    args = parser.parse_args()

    model_name: str = args.model_name

    # Default ONNX path: models/<model-name-with-slashes-replaced>.onnx
    if args.onnx_path is None:
        safe_name = model_name.replace("/", "-")
        onnx_path = Path("models") / f"{safe_name}.onnx"
    else:
        onnx_path = Path(args.onnx_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    out_csv = Path(f"results/raw/m2_coreml_vs_cpu_{model_name}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model name: {model_name}")
    print(f"[INFO] ONNX path:  {onnx_path}")
    print(f"[INFO] Output CSV: {out_csv}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # You can expand this list later if you want more seq lens / batch sizes
    configs: List[M2ExperimentConfig] = [
        M2ExperimentConfig("cpu_only_s128_b1", "cpu_only", 128, 1),
        M2ExperimentConfig("coreml_cpu_s128_b1", "coreml_plus_cpu", 128, 1),
        M2ExperimentConfig("cpu_only_s512_b1", "cpu_only", 512, 1),
        M2ExperimentConfig("coreml_cpu_s512_b1", "coreml_plus_cpu", 512, 1),
    ]

    file_exists = out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = None
        for cfg in configs:
            row = run_ort_benchmark(
                cfg, onnx_path, tokenizer, model_name, args.verbose_ort
            )
            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
            writer.writerow(row)

    print(f"\n[INFO] Appended results to {out_csv}")


if __name__ == "__main__":
    main()
