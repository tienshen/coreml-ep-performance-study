# 📘 ML Systems Benchmark Suite — Technical Report

### *Characterizing transformer inference performance across CPU and GPU backends*

---

## 1. Abstract

This report presents a systematic evaluation of transformer inference performance across CPU and GPU hardware using ONNX Runtime. We benchmark BERT-base and DistilBERT with controlled variations in batch size, sequence length, and model architecture to understand how compute patterns, memory behavior, and runtime scheduling influence end-to-end latency and throughput.

All experiments were executed on an AMD Ryzen 7 3700X CPU and an NVIDIA RTX 3060 Ti Founders Edition GPU. Results highlight differing compute regimes (compute-bound vs memory-bound), nonlinear scaling behavior with batching, and architectural impacts such as model depth reduction.

This study supports my broader effort to build intuition for ML systems behavior and develop practical skills for performance engineering in AI inference pipelines.

---

## 2. Introduction & Motivation

Transformer-based models behave very differently on various hardware backends. Understanding *why* performance improves or degrades—rather than just observing the output—is a core skill in ML Systems Engineering, AI Infrastructure, and Accelerator Performance roles.

This project investigates:

- CPU vs GPU behavior under transformer workloads  
- Interaction between batch size and kernel efficiency  
- Sequence length scaling and attention’s quadratic cost  
- Architectural differences between BERT-base and DistilBERT  
- How ONNX Runtime schedules operators across devices  

The goal is not to achieve the fastest numbers possible but to gain a practical understanding of **how transformer inference behaves across real hardware**.

---

## 3. Background

### 3.1 Transformer Compute Characteristics

A transformer layer comprises several operations:

- Dense projections for Q, K, V  
- Attention score computation: **O(n² × d)**  
- Softmax over attention scores  
- Context vector computation  
- Feed-forward MLP block  
- LayerNorm and residual pathways  

While attention is mathematically quadratic in sequence length (**n²**), real-world runtimes include significant non-quadratic components that soften this scaling at smaller input sizes.

### 3.2 CPU vs GPU Execution Patterns

**CPU:**
- Limited parallelism  
- Smaller GEMM tiles  
- Sensitive to cache locality  
- Larger overhead from Python-level orchestration  
- Performance tends to reveal algorithmic complexity directly  

**GPU:**
- Large GEMMs mapped to Tensor Cores  
- High warp occupancy at moderate batch/seq sizes  
- Kernel launch overhead dominates for small tensors  
- Memory bandwidth becomes dominant at larger workloads  

These factors lead to different latency curves even under identical workloads.

---

## 4. Experimental Setup

### Hardware
- **CPU:** AMD Ryzen 7 3700X (8-core)
- **GPU:** NVIDIA RTX 3060 Ti Founders Edition (8 GB GDDR6)
- **Memory:** 32 GB
- **OS:** Windows 10

### Software Stack
- Python 3.9  
- ONNX Runtime CPUExecutionProvider  
- ONNX Runtime CUDAExecutionProvider  
- HuggingFace Transformers (for ONNX export)  
- NumPy, Matplotlib  

### Models
- **BERT-base-uncased** (12-layer transformer encoder)  
- **DistilBERT-base-uncased** (6-layer distilled encoder)

Both were exported to ONNX using opsets compatible with ORT CUDA EP.

### Benchmark Methodology

Each benchmark run:

- Executes **100 inferences**  
- Reports mean latency, throughput, p50/p90/p99  
- Uses synthetic tokenized input (input_ids, attention_mask, token_type_ids)  
- Measures end-to-end inference latency through ONNX Runtime  

All results saved in JSON and plotted via Python.

---

## 5. Results

### 5.1 Baseline Latency (batch=1, seq_len=128)

| Model | CPU Latency | GPU Latency |
|--------|------------|------------|
| **BERT-base** | ~67–73 ms | ~4.1 ms |
| **DistilBERT** | **33.4 ms** | **2.1 ms** |

**Observation:**  
DistilBERT is ~2× faster than BERT on both CPU and GPU.  
This matches the architectural reduction from **12 → 6 layers**.

---

### 5.2 Batch Size Scaling (BERT-base)

![Batch Scaling Plot](results/plots/bert-base-uncased_cpu_gpu_throughput_batch_sweep.png)

*Figure 1 — Throughput vs batch size. GPU exhibits a non-monotonic dip at batch=8 due to kernel fusion behaviors and CPU-offloaded ops, followed by saturation at batch=32.*

GPU throughput improves with batch size but shows a **non-monotonic dip**:

- batch=1 → ~241 inf/s  
- batch=4 → increases  
- **batch=8 → decreases**  
- batch=32 → ~461 inf/s (peak)

**Interpretation:**  
Batch=8 lies in a transition zone where:

- kernels are large enough to incur memory traffic
- but not large enough to exploit full GPU parallelism  
- ONNX Runtime schedules some ops on CPU (shape ops, small elementwise ops)

This is consistent with ONNX Runtime’s documented scheduling behavior.

---

### 5.3 Sequence Length Scaling (BERT-base)

![Sequence Scaling Plot](results/plots/bert-base-uncased_cpu_gpu_seq_scaling_bs1.png)

*Figure 2 — Latency vs sequence length (batch=1) for CPU (Ryzen 3700X) and GPU (RTX 3060 Ti).

| seq_len | CPU Latency | GPU Latency |
|--------|-------------|-------------|
| 64  | 36.4 ms | 3.65 ms |
| 128 | 67.5 ms | 3.91 ms |
| 256 | 142.7 ms | 6.04 ms |
| 384 | 230.3 ms | 8.65 ms |

**CPU:** Grows super-linearly, almost quadratic as attention’s quadratic cost begins to dominate.  
**GPU:** Initially flat (launch overhead dominated), then grows moderately as matrices enlarge.

The CPU–GPU performance gap increases from **10×** at seq=64 to **26×** at seq=384.

---

## 6. Analysis

### 6.1 CPU Behavior
- CPU performance reveals algorithmic complexity directly  
- Cache locality worsens as sequence length increases  
- GEMM tiles remain small  
- Elementwise and residual ops accumulate overhead  
- Clear superlinear growth emerges past seq_len ≈ 128  

### 6.2 GPU Behavior
- Small sequence lengths underutilize the GPU  
- Latency plateaus until attention matrices reach a meaningful size  
- Larger sequences improve warp occupancy and tiling  
- Eventually becomes a mix of compute-bound and memory-bound  
- Overall growth remains smooth and sub-quadratic at tested sizes  

### 6.3 DistilBERT vs BERT
Latency reduction mirrors the model depth reduction:

- Depth: **12 → 6**
- Latency: **~67 ms → 33 ms (CPU)**  
- Latency: **~4.1 ms → 2 ms (GPU)**  

This confirms that transformer encoder depth contributes roughly linearly to inference cost in this setting.

---

## 7. Discussion

This benchmark suite surfaces several important ML systems principles:

- **Attention’s true complexity** only emerges at larger sequence lengths  
- **Batch scaling is architecture-dependent**, especially on GPUs  
- **Kernel launch overhead matters** for small inputs  
- **Runtime scheduling (ORT)** can meaningfully influence performance  
- **Model architecture directly affects compute demand**  
- **Hardware characteristics (compute throughput, memory bandwidth)** shape scaling curves  

These insights reflect real constraints in production inference systems where throughput, latency, and resource utilization must be understood holistically.

---

## 8. Conclusion

This project demonstrates foundational ML systems engineering skills:

- Designing controlled performance experiments  
- Measuring and interpreting CPU vs GPU behavior  
- Understanding transformer model dynamics  
- Identifying bottlenecks and explaining scaling patterns  
- Connecting model architecture to real hardware performance  
- Working with ONNX Runtime’s execution providers  

These experiments provide practical intuition for future work involving quantization, TensorRT optimization, and LLM inference performance.

---

## 9. Future Work

Planned extensions:

- Throughput vs sequence length plotting  
- TensorRT FP16 and INT8 benchmarking  
- Model-vs-model comparison graphs  
- Add ResNet/Vision Transformer for multimodal coverage  
- Integrate M2 backend  
- Add a small LLM (TinyLlama / GPT2-small)  
- Operator-level profiling (attention Q/K/V breakdown)  
- Add CLI + config interface  

---


## Installation

1. Clone the repository:
```bash
git clone <https://github.com/tienshen/ml-systems-bench.git>
cd ml-systems-bench
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional: CUDA Support

For GPU benchmarking, install ONNX Runtime with CUDA support:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Note: Requires CUDA toolkit to be installed on your system.

## Quick Start

### 1. Export a Model to ONNX

Export a HuggingFace model to ONNX format:

```bash
python scripts/export_to_onnx.py bert-base-uncased
```

Options:
- `--output-name`: Custom output filename
- `--max-length`: Maximum sequence length (default: 128)
- `--opset-version`: ONNX opset version (default: 14)
- `--cache-dir`: HuggingFace cache directory

Example with options:
```bash
python scripts/export_to_onnx.py distilbert-base-uncased \
    --output-name distilbert.onnx \
    --max-length 256
```

### 2. Run Benchmarks

Run benchmarks on an exported model:

```bash
python scripts/run_benchmarks.py --model bert-base-uncased
```

Options:
- `--device`: Device to use (`cpu` or `cuda`)
- `--num-iterations`: Number of benchmark iterations (default: 100)
- `--warmup-iterations`: Number of warmup iterations (default: 5)
- `--batch-size`: Input batch size (default: 1)
- `--seq-length`: Input sequence length (default: 128)
- `--num-threads`: Number of CPU threads (default: auto)
- `--output`: Output file for results

Example:
```bash
python scripts/run_benchmarks.py \
    --model bert-base-uncased \
    --device cpu \
    --num-iterations 100 \
    --batch-size 1 \
    --seq-length 128 \
    --num-threads 4
```

### 3. List Available Models

```bash
python scripts/run_benchmarks.py --list-models
```

## Usage Examples

### Benchmark CPU vs CUDA

```bash
# Benchmark on CPU
python scripts/run_benchmarks.py --model bert-base-uncased --device cpu

# Benchmark on CUDA (requires onnxruntime-gpu)
python scripts/run_benchmarks.py --model bert-base-uncased --device cuda
```

### Custom Batch Sizes and Sequence Lengths

```bash
python scripts/run_benchmarks.py \
    --model distilbert-base-uncased \
    --batch-size 8 \
    --seq-length 256
```

### Different Thread Counts

```bash
# Single thread
python scripts/run_benchmarks.py --model bert-base-uncased --num-threads 1

# Multiple threads
python scripts/run_benchmarks.py --model bert-base-uncased --num-threads 8
```

## Benchmark Metrics

The framework collects the following metrics:

- **Latency**: Per-sample inference time (mean, min, max)
- **Throughput**: Samples processed per second
- **Memory Usage**: Memory consumption during inference
- **Device Information**: Device type and configuration

Results are saved as JSON files in `results/raw/` with timestamps.

## Extending the Framework

### Adding a New Backend

1. Create a new file in `bench/backends/` (e.g., `openvino_runner.py`)
2. Inherit from `BaseRunner`
3. Implement `load_model()` and `run_inference()` methods

Example:
```python
from .base_runner import BaseRunner

class OpenVINORunner(BaseRunner):
    def __init__(self, model_path: str):
        super().__init__(model_path, "openvino")
    
    def load_model(self):
        # Implementation
        pass
    
    def run_inference(self, input_data):
        # Implementation
        pass
```

### Adding Custom Metrics

Extend the `BenchmarkMetrics` class in `bench/metrics.py` to add custom metrics collection and analysis.

## Roadmap

- [ ] Add visualization utilities in `plotting.py`
- [ ] Support for more backend types (OpenVINO, TensorRT)
- [ ] Multi-GPU benchmarking
- [ ] Automated comparison reports
- [ ] Power consumption measurements
- [ ] Integration with MLflow for experiment tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ml_systems_bench,
  title = {ML Systems Bench: LLM Performance Benchmarking Framework},
  author = {Tien Shen},
  year = {2025},
  url = {https://github.com/tienshen/ml-systems-bench.git}
}
```

## Troubleshooting

### ONNX Export Issues

If you encounter errors during export:
- Ensure the model is supported by ONNX
- Try a different opset version
- Check PyTorch and ONNX versions are compatible

### CUDA Not Available

If CUDA benchmarks fail:
- Verify CUDA toolkit is installed
- Install `onnxruntime-gpu` instead of `onnxruntime`
- Check GPU drivers are up to date

### Memory Issues

For large models:
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing (if applicable)

## Contact

For questions or issues, please open an issue on GitHub.
