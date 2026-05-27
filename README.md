# fast-rl-grpo


**A performance-obsessed GRPO training pipeline from algorithm to bare-metal CUDA kernels.**

> Optimizing an RL post-training pipeline for LLMs using Group Relative Policy Optimization (GRPO), custom fused cross-entropy CUDA kernels, and systematic Nsight Systems profiling. Built on Llama 3.2-1B, trained on GSM8K, and benchmarked on an L4 GPU.

---

## Table of Contents

- [Overview](#overview)
- [Why GRPO over PPO](#why-grpo-over-ppo)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Performance Journey](#performance-journey)
- [Custom Fused Cross-Entropy Kernel](#custom-fused-cross-entropy-kernel)
- [Getting Started](#getting-started)
- [Profiling & Benchmarking](#profiling--benchmarking)
- [Results Summary](#results-summary)
- [Further Work](#further-work)
- [References](#references)

---

## Overview

This repository implements a complete RL post-training pipeline using **GRPO** (Group Relative Policy Optimization), introduced in the [DeepSeek-Math paper](https://arxiv.org/pdf/2402.03300). The focus is not just on algorithmic correctness, but on _squeezing every millisecond_ out of the training loop through:

- **Custom CUDA kernels**: A fused cross-entropy forward/backward kernel with online softmax and vectorized `float4` loads, outperforming both PyTorch native and Unsloth's Triton implementation.
- **Systematic profiling**: Five iterative versions (`v1` → `v5`), each guided by NVIDIA Nsight Systems traces to identify and eliminate real bottlenecks.
- **Low-level memory optimization**: Pinned memory, non-blocking transfers, TF32 matmul precision, and elimination of non-contiguous memory access patterns.

The pipeline is trained on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) using [Llama 3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) with structured `<think>` / `<answer>` reasoning.

---

## Why GRPO over PPO

Standard **PPO** (Proximal Policy Optimization) requires a learned **Value Function** (critic) of comparable scale to the policy model. For a 70B policy, that means hosting ~two 70B models in VRAM just for training. Additionally, the critic introduces noisy baselines and per-token overhead.

**GRPO** eliminates the critic entirely. Instead, it:

1. Samples a **group** of `G` completions per prompt from the old policy.
2. Scores them with a reward model.
3. Computes **advantages** from the _relative_ rewards within the group (group-relative baseline).
4. Optimizes a clipped surrogate objective with KL regularization against a frozen reference model.

This reduces VRAM requirements to **two** model copies (reference + learnable policy), while the old policy is implicitly captured via pre-computed log-probabilities and advantages before each GRPO inner loop.

**Objective:**

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(\frac{\pi_\theta}{\pi_{\theta_{old}}} \hat{A}_{i,t},\;\text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}},1\pm\varepsilon\right)\hat{A}_{i,t}\right) - \beta \cdot D_{KL}\right)\right]$$

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │  GSM8K      │───▶│  Generation  │───▶│  Reward Functions   │  │
│  │  Dataset    │    │  (Old Policy)│    │  (Correctness +     │  │
│  └─────────────┘    └──────────────┘    │   Format)           │  │
│                            │            └──────────┬──────────┘  │
│                            ▼                       │             │
│                     ┌──────────────┐               │             │
│                     │  Reference   │               │             │
│                     │  Model       │◀──────────────┘             │
│                     │  (Frozen)    │                             │
│                     └──────┬───────┘                             │
│                            │ ref_logps                           │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    GRPO Inner Loop (M iterations)           │ │
│  │                                                             │ │
│  │  Forward ──▶ FusedCrossEntropy ──▶ KL + Clipped Loss       │ │
│  │  Backward ◀── Autograd ◀── FusedCE Backward Kernel         │ │
│  │  Optimizer Step (Fused AdamW)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Two model copies only | Old policy is captured via pre-computed `gen_logps` and advantages no third set of weights needed |
| `torch.compile()` on policy | Fuses fragmented PyTorch ops into efficient Triton kernels for forward/backward |
| Fused AdamW optimizer | Reduces HBM round-trips by performing param update in a single kernel |
| Custom CUDA cross-entropy | Eliminates `nn.CrossEntropyLoss` transpose overhead and non-contiguous memory patterns |
| Pinned memory + `non_blocking` | DMA transfers from page-locked CPU memory remove OS-level staging overhead |
| TF32 matmul precision | Keeps FP32 exponent range but trims mantissa to 10 bits massive throughput gain on Ampere+ |

---

## Repository Structure

```
fast-rl-grpo/
├── rl.py                          # Full training pipeline (final optimized version)
├── requirements.txt               # Python dependencies
│
├── perf/                          # Iterative optimization versions
│   ├── v1.py                      #   Naive baseline
│   ├── v2.py                      #   + torch.compile, Fused AdamW, nn.CrossEntropyLoss
│   ├── v3.py                      #   + .reshape() fix, TF32
│   ├── v4.py                      #   + Custom fused cross-entropy kernel
│   └── v5.py                      #   + Pinned memory, non-blocking transfers
│
├── cross_entropy_kernels/         # Kernel development iterations (standalone .cpp)
│   ├── ce_fwd_1.cpp               #   Forward v1: branchless warp reduction
│   ├── ce_fwd_2.cpp               #   Forward v2: branched warp reduction
│   ├── ce_fwd_3.cpp               #   Forward v3: vectorized float4 loads
│   ├── ce_fwd_4.cpp               #   Forward v4: + LSE output for backward
│   ├── ce_bwd_1.cpp               #   Backward v1: vectorized loads, single block
│   ├── ce_bwd_2.cpp               #   Backward v2: 2D grid, chunked vocab
│   └── ce_bwd_3.cpp               #   Backward v3: final integrated version
│
├── cuda_src/                      # Production CUDA extension (installable)
│   ├── cross_entropy.cu           #   Final fused forward + backward kernels
│   ├── bindings.cpp               #   pybind11 bindings
│   ├── fused_ce.py                #   torch.library + autograd.Function wrapper
│   └── setup.py                   #   setuptools build (pip install .)
│
├── benchmark/                     # Kernel benchmarking scripts
│   ├── bench_ce_fwd.py            #   Forward kernel: Custom vs PyTorch vs Unsloth
│   ├── bench_ce_bwd.py            #   Backward kernel: Custom vs PyTorch vs Unsloth
│   └── bench_fused_ce.py          #   Full fused CE (fwd + bwd) benchmark
│
├── perf_profiling/                # Nsight Systems profiling artifacts
│   ├── nsys_profiling.zip         #   Nsight .nsys-rep files (viewable in Nsight GUI)
│   ├── ce_fused_trace.json        #   torch.profiler traces (fused kernel)
│   └── ce_trace_*.json            #   torch.profiler traces (kernel iterations)
│
└── utilities/
    └── profiling.txt              # GPU env setup & nsys invocation commands
```

---

## Performance Journey

Each version was profiled with **NVIDIA Nsight Systems** using NVTX-annotated forward/backward/optimizer phases. GPU clocks were locked at ~88% peak (1800 MHz on L4) for reproducible measurements.

### Optimization Timeline

| Version | Avg Step Time | Forward | Backward | Optimizer | Key Changes |
|---------|--------------|---------|----------|-----------|-------------|
| **v1** | 670 ms | 144 ms | 315 ms | 207 ms | Naive baseline (manual softmax + gather loop) |
| **v2** | 747 ms  | 278 ms | 322 ms | 148 ms | `torch.compile`, Fused AdamW, `nn.CrossEntropyLoss` _regression from transpose_ |
| **v3** | 526 ms | 116 ms | 263 ms | 147 ms | `.reshape()` fix for contiguous memory, TF32 enabled |
| **v4** | 518 ms | 113 ms | 260 ms | 145 ms | Custom fused cross-entropy CUDA kernel |
| **v5** | ~518 ms | 113 ms | 260 ms | 145 ms | Pinned memory + `non_blocking` transfers _eliminated GPU idle gaps_ |

**Total speedup: ~1.29× (666 ms → 518 ms per GRPO step)**

### Key Insights

1. **v2 regression**: `nn.CrossEntropyLoss` expects `(N, C, ...)` input. Using `.transpose(1,2)` on a `(B, L, V)` tensor creates non-contiguous memory access, causing a **172 ms** cross-entropy call vs the original **20 ms**. Fix: use `.reshape(-1, V)` which copies to contiguous memory when needed.

2. **Optimizer is memory-bound**: Standard AdamW performs separate read/write cycles per parameter. Fused AdamW loads parameters into registers once, performs all math, and writes back **1.4× speedup (207 ms → 148 ms)**.

3. **Custom kernel ROI**: The fused cross-entropy kernel is **2× faster** than PyTorch native, but cross-entropy is only ~2% of the total step time. The real win came from eliminating the `.reshape()` copy entirely.

4. **Pinned memory**: While total step time didn't change measurably, Nsight traces show GPU idle gaps dropped from **3.6 ms → 0.25 ms**, creating a continuous execution profile essential at scale.

---

## Custom Fused Cross-Entropy Kernel

The custom CUDA kernel is the heart of this project's low-level optimization work.

### Design

**Standard approach** (4-step reduction):
```
Pass 1: Find max logit (warp → block reduce)
Pass 2: Compute sum of exp(logit - max) (warp → block reduce)
Compute: logsumexp = max + log(sum)
Compute: loss = logsumexp - target_logit
```

**Our approach** (2-step, online softmax):
```
Single Pass: Maintain running (max, sum_exp) per thread using online safe-max correction
Reduce: warp → block reduction with compound (max, sum_exp) merging
Compute: loss = logsumexp - target_logit
```

### Kernel Techniques

| Technique | Purpose |
|---|---|
| **Online softmax** | Single-pass LogSumExp collapses two full-vocab sweeps into one |
| **Vectorized `float4` loads** | Each thread loads 8× `bfloat16` values per memory transaction (128-bit) |
| **`__nv_bfloat162` pair unpacking** | Hardware-accelerated bf16 → fp32 conversion for high-precision accumulation |
| **Branched block reduction** | Eliminates redundant `__expf` calls vs branchless no perf difference since kernel is memory-bound |
| **2D grid launch (backward)** | Chunks vocab into 2048-element tiles across blocks maximizes SM utilization for large V |
| **LSE caching** | Forward pass stores LogSumExp values; backward reuses them without recomputation |
| **`#pragma unroll`** | Compiler hint for the inner 4-pair loop reduces loop overhead |

### Benchmark Results (L4 GPU, V=156,000, B×L=256)

| Kernel | Forward | Backward |
|--------|---------|----------|
| PyTorch `nn.CrossEntropyLoss` | 1.41 ms | 1.43 ms |
| Unsloth (Triton) | 0.61 ms | 0.72 ms |
| **Ours (CUDA)** | **0.34 ms** | **0.71 ms** |
| | **4.1× vs PyTorch** | **2.0× vs PyTorch** |

### PyTorch Integration

The kernel integrates cleanly with PyTorch's ecosystem:

```python
# Registration via torch.library (torch.compile compatible)
@torch.library.custom_op("custom_ce::fused_ce_fwd", mutates_args=())
def fused_ce_fwd(logits, labels) -> tuple[torch.Tensor, torch.Tensor]: ...

# Shape inference for torch.compile tracing
@fused_ce_fwd.register_fake
def _(logits, labels): ...

# Autograd integration
class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        loss, lse = torch.ops.custom_ce.fused_ce_fwd(logits, labels)
        ctx.save_for_backward(logits, labels, lse)
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        logits, labels, lse = ctx.saved_tensors
        grad_logits = torch.ops.custom_ce.fused_ce_bwd(grad_outputs[0], logits, lse, labels)
        return grad_logits, None
```

---

## Getting Started

### Prerequisites

- **GPU**: NVIDIA Ampere+ (L4, A100, H100) with CUDA 12.4+
- **Python**: 3.10+
- **CUDA Toolkit**: 12.4+ (for kernel compilation)
- **Hugging Face Token**: Required for Llama 3.2 model access

### Installation

```bash
# Clone
git clone https://github.com/sagar0x0/fast-rl-grpo.git
cd fast-rl-grpo

# Install Python dependencies
pip install -r requirements.txt

# Build & install the custom CUDA cross-entropy kernel
cd cuda_src
pip install .
cd ..
```

### Training

```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Run the full optimized training pipeline
python rl.py
```

### Configuration

Key hyperparameters in `rl.py`:

| Parameter | Default | Description |
|---|---|---|
| `Q_batch_size` | 1 | Number of questions per batch |
| `num_gen_Q` | 4 | Group size G (completions per prompt) |
| `model_path` | `meta-llama/Llama-3.2-1B-Instruct` | Base model |
| `max_prompt_length` | 300 | Max input token length |
| `max_new_token_len` | 256 | Max generation length |
| `num_iterations` | 4 | GRPO inner loop iterations per step |
| `lr` | 5e-7 | Learning rate |
| `beta` | 0.04 | KL penalty coefficient |
| `clip_param` | 0.2 | PPO-style clipping parameter |
| `all_steps` | 500 | Total training steps |

### Reward Functions

The pipeline uses two reward signals, summed per completion:

- **Correctness reward** (`±1.0`): Extracts the final numeric answer and compares against GSM8K ground truth.
- **Format reward** (`±1.0`): Checks for proper `<think>...</think><answer>...</answer>` structure.

---

## Profiling & Benchmarking

### GPU Environment Setup

```bash
# Lock GPU clocks for stable benchmarking (80-90% of max)
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1800,1800

# Reset after benchmarking
sudo nvidia-smi -rgc
```

### Nsight Systems Profiling

```bash
# Install Nsight Systems
sudo apt install nsight-systems-2025.5.2
export PATH=/opt/nvidia/nsight-systems/2025.5.2/bin:$PATH

# Profile a training run
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=cpu \
  --output=rl_profile \
  --force-overwrite=true \
  --stats=true \
  python rl.py
```

The profiling code uses NVTX ranges to label each phase:

```python
with torch.autograd.profiler.emit_nvtx():
    for i in range(4):
        nvtx.range_push(f"Step_{i}")
        with record_function("forward_pass"):
            loss = GRPO_step(batch)
        with record_function("backward_pass"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()
        nvtx.range_pop()
```

### Kernel Benchmarks

```bash
# Forward kernel benchmark (Custom vs PyTorch vs Unsloth)
python benchmark/bench_ce_fwd.py

# Backward kernel benchmark
python benchmark/bench_ce_bwd.py

# Full integrated fwd+bwd benchmark
python benchmark/bench_fused_ce.py
```

---

## Results Summary

```
Pipeline Optimization                     Kernel Benchmarks (forward, V=156K)
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│  v1 (naive)          670 ms      │      │  PyTorch native      1.41 ms    │
│  v2 (compile+fused)  747 ms ▲    │      │  Unsloth (Triton)    0.61 ms    │
│  v3 (reshape+TF32)   526 ms ▼▼  │      │  Ours (CUDA)         0.34 ms ★  │
│  v4 (custom kernel)  518 ms ▼   │      │                                  │
│  v5 (pinned mem)    ~518 ms ═   │      │  4.1× faster than PyTorch        │
│                                  │      │  1.8× faster than Unsloth        │
│  Total: 1.29× speedup           │      └──────────────────────────────────┘
└──────────────────────────────────┘
```

---

## Further Work

- **`torch.compile` over full `GRPO_step`**: Force graph-level fusion across the entire training iteration, not just the model forward pass.
- **vLLM inference engine**: Offload the generation/sampling phase to a dedicated inference server, decoupling generation throughput from training compute.
- **Multi-GPU / FSDP**: Scale beyond single-GPU with fully sharded data parallelism.
- **Chunked cross-entropy**: Process the vocabulary in memory-efficient chunks to reduce peak VRAM during the loss computation for larger models.

---

## References

- **GRPO**: [DeepSeek-Math: Integrating Mathematical Reasoning in LLMs](https://arxiv.org/pdf/2402.03300)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **Online Softmax**: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- **Blog Post**: [RL Training with GRPO: Kernels & Performance](https://sagar0x0.substack.com/p/rl-training-with-grpo-kernels-and)

---

## License

MIT
