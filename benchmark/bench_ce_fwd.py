import os 
import torch
from torch.utils.cpp_extension import load_inline
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
import unsloth

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"    # for t4 gpu

kernel_cpp = r"""
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
// #include <torch/extension.h>

__device__ __forceinline__ void warp_reduce_online(float &m, float &d) {
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_m = __shfl_down_sync(0xffffffff, m, offset);
    float other_d = __shfl_down_sync(0xffffffff, d, offset);

    float prev_m = m;
    m = fmaxf(m, other_m);
    d = d * (__expf(prev_m - m)) + other_d * (__expf(other_m - m));
  }
}

__device__ __forceinline__ void block_reduce_online(float &m, float &d) {
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_m = __shfl_down_sync(0xffffffff, m, offset);
    float other_d = __shfl_down_sync(0xffffffff, d, offset);

    if (other_m > m) {
      // Case: The new value is larger
      // d * exp(m - other_m) is safe because (m - other_m) is negative
      d = d * __expf(m - other_m) + other_d;
      m = other_m;
    } else if (other_m != -INFINITY) {
      // Case: m is larger or equal, and other_m is a real number
      // other_d * exp(other_m - m) is safe because (other_m - m) is negative
      d += other_d * __expf(other_m - m);
    }
  }
}

__global__ void
cross_entropy_optimized_kernel(const __nv_bfloat16 *__restrict__ logits,
                               const int64_t *__restrict__ labels,
                               __nv_bfloat16 *__restrict__ loss,
                               int64_t logits_row_stride) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int num_warps = blockDim.x / 32;

  const __nv_bfloat16 *row_logits = logits + batch_idx * logits_row_stride;
  const int64_t label = labels[batch_idx];

  extern __shared__ unsigned char smem[];
  // float is accumulator scalar type
  float *s_m = reinterpret_cast<float *>(smem);
  float *s_d = s_m + num_warps;

  float m = -INFINITY;
  float d = 0.0f;

  // local register accumulation (colaesced memory loads)
  for (int i = tid; i < logits_row_stride; i += blockDim.x) {
    float row_logit =
        __bfloat162float(row_logits[i]); // explicit scalar_t to acc_scalar_t
    float m_prev = m;
    m = fmaxf(m_prev, row_logit);
    // branchless programming
    d = d * (__expf(m_prev - m)) + __expf(row_logit - m);
  }

  // warp reduction
  warp_reduce_online(m, d);

  // warp reduce to smem
  if (lane_id == 0) {
    s_m[warp_id] = m;
    s_d[warp_id] = d;
  }
  __syncthreads();

  // block reduction  (given warp_size = 32, even as low as 48 thread_block
  // handled)
  if (warp_id == 0) {
    m = (tid < num_warps) ? s_m[lane_id] : -INFINITY;
    d = (tid < num_warps) ? s_d[lane_id] : 0.0f;

    block_reduce_online(m, d);
  }

  // final loss calculation

  if (tid == 0) {
    float log_sum_exp = logf(d) + m;
    float target_logit = __bfloat162float(row_logits[label]);
    loss[batch_idx] = __float2bfloat16(log_sum_exp - target_logit);
  }
}

void cross_entropy_optimized_launch(uintptr_t logits_ptr, // logits [B*L, V]
                                    int64_t logits_row_stride, // V
                                    uintptr_t loss_ptr,        // [B*L]
                                    uintptr_t labels_ptr,      // [B*L]
                                    int rows                   // B*L
) {
  auto *logits = reinterpret_cast<__nv_bfloat16 *>(logits_ptr);
  auto *loss = reinterpret_cast<__nv_bfloat16 *>(loss_ptr);
  auto *labels = reinterpret_cast<int64_t *>(labels_ptr);

  // Optimal configuration - tune based on your hardware
  const int block_size = 256; // 8 warps
  const int num_warps = block_size / 32;
  const size_t shared_mem_size =
      2 * num_warps * sizeof(float); // float is acc scalar type

  dim3 grid(rows);
  dim3 block(block_size);

  cross_entropy_optimized_kernel<<<grid, block, shared_mem_size>>>(
      logits, labels, loss, logits_row_stride);

  // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(cross_entropy_kernel, m) {
  m.def("cross_entropy_kernel", &cross_entropy_optimized_launch,
        "Cuda cross_entropy kernel");
}
"""

ce_module = load_inline(
    name = "cross_entropy_kernel",
    cpp_sources="",
    cuda_sources=kernel_cpp ,
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-std=c++20"]
)

def check_ce():
    logits = torch.rand([256, 156000], dtype=torch.bfloat16, device='cuda')
    input_ids = torch.randint(0, 156000, (256, ), dtype=torch.int64, device='cuda')

    num_rows = logits.shape[0]
    row_stride = logits.shape[1]   # V   : python int are 64bits by default

    loss_custom = torch.zeros([256], dtype=torch.bfloat16, device='cuda')
    ce_module.cross_entropy_kernel(
        logits.data_ptr(),
        row_stride,
        loss_custom.data_ptr(),
        input_ids.data_ptr(),
        num_rows
    )

    torch.cuda.synchronize()
    
    ce = nn.CrossEntropyLoss(reduction='none')
    loss_baseline = ce(logits, input_ids)

    # max abs diff
    max_diff = torch.max(torch.abs(loss_custom - loss_baseline))
    print(f"Max Absolute Difference: {max_diff}")

    # torch.isclose chech: atol and rtol relaxed for bfloat16
    is_corr = torch.isclose(loss_custom, loss_baseline, rtol=1e-2, atol=1e-3)
    print(f"Is Close: {is_corr}")

def bench_ce():
    # setup
    B, V = 256, 156000
    logits = torch.rand([B, V], dtype=torch.bfloat16, device='cuda')
    input_ids = torch.randint(0, V, (B, ), dtype=torch.int64, device='cuda')

    num_rows = logits.shape[0]
    row_stride = logits.shape[1]   # V   : python int are 64bits by default

    loss_custom = torch.zeros([B], dtype=torch.bfloat16, device='cuda')

    # Warm-up (Critical)
    for _ in range(20):
        ce_module.cross_entropy_kernel(logits.data_ptr(), row_stride, loss_custom.data_ptr(), input_ids.data_ptr(), num_rows)
        nn.CrossEntropyLoss(reduction='none')(logits, input_ids)
        unsloth.kernels.cross_entropy_loss.Fast_CrossEntropyLoss.apply(logits, input_ids)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("custom_ce"):
            ce_module.cross_entropy_kernel(
                logits.data_ptr(),
                row_stride,
                loss_custom.data_ptr(),
                input_ids.data_ptr(),
                num_rows
            )
        
        with record_function("pytorch_ce"):
            loss_baseline = nn.CrossEntropyLoss(reduction='none')(logits, input_ids)
        
        with record_function("unsloth_triton"):
            unsloth_loss = unsloth.kernels.cross_entropy_loss.Fast_CrossEntropyLoss.apply(logits, input_ids)

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    # chrome tracing
    prof.export_chrome_trace("ce_trace.json")




if __name__ == '__main__':
    if ce_module:
        bench_ce()