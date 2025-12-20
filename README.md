# 📘 CoreML Execution Provider: Systems-Level Performance Study

### Diagnosing inference performance, graph partitioning, and accelerator offload on Apple Silicon

---

## 1. Abstract

This project is a systems-level investigation into **inference performance on Apple Silicon** using **ONNX Runtime (ORT) with the CoreML Execution Provider (EP)**. Rather than benchmarking raw speedups, the goal is to understand *why* CoreML acceleration succeeds or fails across different models, dtypes, and graph structures.

Through controlled experiments on small Transformer models and a vision control model (MobileNet), this work characterizes:

- graph partitioning and fragmentation behavior  
- CPU fallback mechanisms  
- frontend dtype compatibility  
- dispatch and transition overhead  
- conditions under which CoreML provides real acceleration  

The results show that **frontend compatibility and graph structure dominate performance**, often outweighing theoretical hardware advantages of the Apple Neural Engine (ANE).

---

## 2. Motivation

Apple’s CoreML stack is widely used for deploying ML models on edge devices, yet developers frequently encounter confusing performance outcomes:

- CoreML sometimes underperforms CPU execution  
- FP16 models can be slower than FP32  
- Accelerators appear enabled but provide little benefit  
- Performance changes dramatically with minor graph edits  

This project addresses a practical question:

> **When should CoreML be used for inference — and when should it not?**

Rather than treating CoreML as a black box, we use **profiling, graph analysis, and controlled ablations** to surface the mechanisms that govern runtime behavior.

---

## 3. Experimental Scope

### Models
- **Transformer family**
  - BERT-derived small Transformer variants
  - Focus on batch-1 / latency-sensitive inference
- **Vision control**
  - MobileNet (convolutional workload)

### Runtime
- ONNX Runtime  
  - CPUExecutionProvider  
  - CoreMLExecutionProvider (with CPU fallback)

### Hardware
- Apple Silicon (M-series)  
  - CPU  
  - CoreML backend (ANE / GPU selected internally)

### Profiling
- ORT JSON profiler  
- Per-node kernel execution times  
- Partition count and provider assignment  

---

## 4. Key Observations

### 4.1 CoreML Performance Anomaly

First, we benchmarked the BERT-base-uncased transformer on CPU vs CoreML and found CPU outperformed CoreML by a wide margin.

![BERT-base vs CPU/CoreML](results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_bert-base-uncased_combined.png)

BERT has roughly:
- 110M parameters
- 12 transformer encoder layers
- Hidden size 768, Vocabulary size ~30k

Initial profiling reveals:
- heavy graph fragmentation  
- frequent CPU↔CoreML transitions  
- significant dispatch overhead  

Could the heavy graph fragmentation and frequent CPU↔CoreML transitions be caused by transformer model size exceeding ANE unsupported embedded matrix size or unsupported transformer encoder layers?

Much of the CoreML architecture and specs are propriatary and not made publicly available. We can only obtain system insights by stress testing CoreML.

![DistilBERT vs CPU/CoreML](results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_distilbert-base-uncased_combined.png)

DistilBERT has roughly:
- ~66M parameters
- 6 transformer layers (vs 12)
- Hidden size 768

![tiny-systems-bert vs CPU/CoreML](results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_tiny-systems-bert_combined.png)

Tiny Systems Bert has roughly:
  - ~4.4M parameters
  - 2 transformer layers
  - Hidden size 128

Our benchmark results across different models shows that latency and throughput improved from utilizing models with reduced layers or hidden size. However, CoreML consistently underperformed CPU by a wide margin even when model size was reduced.

This motivates a deeper diagnosis.

---

### 4.2 Graph Fragmentation, Not Memory, Is the Bottleneck

ORT partitions the ONNX graph into multiple CoreML subgraphs separated by CPU-only operators.

In the `tiny-systems-bert` FP32 dynamic-batch experiment, profiling recorded **20 CoreML partitions interleaved with CPU nodes**. The trace (2.96 s total) shows `SequentialExecutor::Execute` spending ~841 ms just coordinating partition hops, while CPU-only ops such as `/encoder/*/intermediate_act_fn/Erf` (~20 ms per layer), `Where`, `Cast`, and `Expand` saturate the top of the kernel list. This concretes the fragmentation diagnosis: each CoreML block is short-lived and immediately followed by small CPU kernels.

Common cut-makers include:
- `Erf` (from GELU)  
- `Where`  
- `Cast`  
- `Expand`  
- `Unsqueeze`  

Each partition boundary introduces:
- synchronization cost  
- dispatch overhead  
- loss of kernel fusion opportunities  

This explains why **batching improves throughput without reducing fragmentation**: overhead is amortized, not removed.

[Full profile summary](results/txt/tiny-systems-bert_fp32_dynamic_gelu_profile_summary.txt)

---

### 4.3 Static Shapes Are Mandatory for Stable Behavior

Dynamic shapes lead to:
- unstable partitioning  
- excessive fallback  
- noisy profiling results  

Exporting ONNX models with **fixed (batch, seq_len)**:
- stabilizes execution  
- makes profiling repeatable  
- improves performance consistency  

Static shapes act as **hard contracts** for CoreML EP.

---

### 4.4 Targeted Graph Intervention: GELU → FastGELU

Profiling identifies `Erf` as a dominant CPU cut-maker.

Replacing all GELU instances with **FastGELU**:
- removes `Erf` from the graph  
- reduces CoreML partition count  
- decreases CPU kernel time  
- yields measurable throughput improvement  

This demonstrates that **small, targeted graph edits can outperform generic tuning**.

---

### 4.5 Dtype Pitfall: FP16 Can Disable Acceleration

Contrary to intuition:

- **FP16 ONNX models often fail CoreML partitioning**  
- Execution silently falls back to CPU  
- Partition count may decrease *only because offload disappears*  

In contrast:
- **FP32 ONNX graphs are more reliably ingested**  
- CoreML EP internally lowers precision as needed  
- Full accelerator offload becomes possible  

This shows that **user-visible dtype ≠ execution dtype** in CoreML.

---

### 4.6 Control Experiment: MobileNet on CoreML

To verify that CoreML is not inherently inferior, we benchmark MobileNet:

- FP32 ONNX → 100% CoreML partition  
- No fragmentation  
- ~16× speedup over CPU  

This confirms:
- CoreML excels on workloads with strong op coverage  
- Performance failures in Transformers are **structural**, not hardware limitations  

---

## 5. Key Takeaways for Edge Deployment

For developers deploying ML on Apple devices:

- **Do not assume FP16 is faster**  
- **Measure offload coverage**, not just latency  
- **Static shapes are critical** for production inference  
- **Accelerators are not free** — dispatch overhead matters  
- **CPU execution may be the correct choice** for some NLP workloads  
- **Vision models benefit far more reliably** from CoreML  

---

## 6. Repository Layout



---

## 7. Reproducibility

The repository includes:
- representative profiler traces  
- summarized results  
- scripts to regenerate figures  

Due to hardware dependence, **full reproduction requires Apple Silicon**, but analysis scripts are portable.

---

## 8. Conclusion

This project reframes CoreML performance from a “speedup problem” into a **systems diagnosis problem**.

The central insight is that **frontend compatibility and graph structure govern accelerator effectiveness** far more than raw compute capability. By understanding these boundaries, developers can make informed decisions about when CoreML will help — and when it will not.
