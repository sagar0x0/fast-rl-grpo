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

__global__ void
cross_entropy_backward_kernel(const __nv_bfloat16 *__restrict__ logits,
                              const int64_t *__restrict__ labels,
                              const float *__restrict__ logsumexp_buff,
                              __nv_bfloat16 *__restrict__ grad_logits,
                              int64_t logits_row_stride) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane_id = tid / 32;
  const int warp_id = tid % 32;

  const __nv_bfloat16 *row_logits = logits + batch_idx * logits_row_stride;
  __nv_bfloat16 *row_grads = grad_logits + batch_idx * logits_row_stride;

  // logsumexp for this row
  float logsumexp = logsumexp_buff[batch_idx];
  // target input_ids yi : for target == 1, for non-target == 0
  int64_t target_label = labels[batch_idx];

  // vectorized loads
  const float4 *vec_logits = reinterpret_cast<const float4 *>(row_logits);
  float4 *vec_grad = reinterpret_cast<float4 *>(row_grads);

  /*  int y = (target_label == i) ? 1 : 0;
      float grad_logit = - y + __expf(row_logits[i] - logsumexp);
      grad_logits[batch_idx * logits_row_stride + i] =
     __float2bfloat16(grad_logit);
  */

  int num_vec_ele = logits_row_stride / 8;

  // global mem vectorized load
  for (int i = tid; i < num_vec_ele; i += blockDim.x) {
    float4 vec_bits = vec_logits[i];
    __nv_bfloat162 *logit_pairs = reinterpret_cast<__nv_bfloat162 *>(&vec_bits);

    float4 out_grad_vec;
    __nv_bfloat162 *grad_pairs =
        reinterpret_cast<__nv_bfloat162 *>(&out_grad_vec);

// 4 pairs (8 elements)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float val1 = __bfloat162float(logit_pairs[j].x);
      float val2 = __bfloat162float(logit_pairs[j].y);

      // compute grad : P - y
      int idx_val1 = i * 8 + j * 2;
      int idx_val2 = i * 8 + j * 2 + 1;

      float grad_val1 =
          __expf(val1 - logsumexp) - (target_label == idx_val1 ? 1.0f : 0.0f);
      float grad_val2 =
          __expf(val2 - logsumexp) - (target_label == idx_val2 ? 1.0f : 0.0f);

      grad_pairs[j] = __floats2bfloat162_rn(grad_val1, grad_val2);
    }

    // store all 8 elements
    vec_grad[i] = out_grad_vec;
  }

  // leftover
  int tail_start = num_vec_ele * 8;

  // Every thread checks if there are leftovers it needs to handle
  for (int i = tail_start + tid; i < logits_row_stride; i += blockDim.x) {
    // Load explicitly as bfloat16
    float val = __bfloat162float(row_logits[i]);
    float grad_val =
        __expf(val - logsumexp) - (target_label == i ? 1.0f : 0.0f);
    
    row_grads[i] = __float2bfloat16(grad_val);
  }
}

void cross_entropy_backward_launch(uintptr_t logits_ptr,      // logits [B*L, V]
                                   int64_t logits_row_stride, // V
                                   uintptr_t logsumexp_buf_ptr, // [B*L]
                                   uintptr_t labels_ptr,        // [B*L]
                                   uintptr_t grad_logit_ptr,    // [B*L, V]
                                   int rows                     // B*L
) {
  auto *logits = reinterpret_cast<__nv_bfloat16 *>(logits_ptr);
  auto *logsumexp_buf = reinterpret_cast<float *>(logsumexp_buf_ptr);
  auto *labels = reinterpret_cast<int64_t *>(labels_ptr);
  auto *grad_logits = reinterpret_cast<__nv_bfloat16 *>(grad_logit_ptr);
  // Optimal configuration - tune based on your hardware
  const int block_size = 256; // 8 warps
  // const int num_warps = block_size / 32;
  // shared memory not used as there is no reduction of threads

  dim3 grid(rows);
  dim3 block(block_size);

  cross_entropy_backward_kernel<<<grid, block>>>(
      logits, labels, logsumexp_buf, grad_logits, logits_row_stride);

  // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(cross_entropy_bckwd_kernel, m) {
  m.def("cross_entropy_bckwd_kernel", &cross_entropy_backward_launch,
        "Cuda cross_entropy backward kernel");
}
"""



ce_module = load_inline(
    name = "cross_entropy_bckwd_kernel",
    cpp_sources="",
    cuda_sources=kernel_cpp,
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-std=c++20"]
)

def check_ce():
    B = 256
    V = 156000
    logits = torch.rand([B, V], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    input_ids = torch.randint(0, V, (B, ), dtype=torch.int64, device='cuda')

    # pytorch reference (fwd + bckwd)
    ce_loss_func = nn.CrossEntropyLoss(reduction='sum') # reduction='sum' to get a scalar output
    loss_baseline = ce_loss_func(logits, input_ids)

    loss_baseline.backward()
    grad_baseline = logits.grad.clone()  # ground truth gradient

    # custom cuda 

    # logsumexp from forward pass
    with torch.no_grad():
        logsumexp = torch.logsumexp(logits.float(), dim=-1)

    grad_custom = torch.zeros([B, V], dtype=torch.bfloat16, device='cuda')

    ce_module.cross_entropy_bckwd_kernel(
        logits.data_ptr(),
        V,
        logsumexp.data_ptr(),
        input_ids.data_ptr(),
        grad_custom.data_ptr(),
        B
    )
    torch.cuda.synchronize()

    # compare gradients

    # max abs diff
    max_diff = torch.max(torch.abs(grad_custom.float() - grad_baseline.float()))
    print(f"Max Absolute Difference: {max_diff}")

    # torch.isclose chech: atol and rtol relaxed for bfloat16
    is_corr = torch.isclose(grad_custom, grad_baseline, rtol=1e-2, atol=1e-3)
    print(f"Is Close: {is_corr}")

    # Check if the sum of gradients is zero (softmax)
    print(f"Sum of custom grads (should be ~0): {grad_custom.sum().item()}")


def bench_ce():
    # setup
    B, V = 256, 156000
    logits = torch.rand([B, V], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    input_ids = torch.randint(0, V, (B, ), dtype=torch.int64, device='cuda')

    grad_custom = torch.zeros([B, V], dtype=torch.bfloat16, device='cuda')

    with torch.no_grad():
        logsumexp = torch.logsumexp(logits.float(), dim=-1)

    ce_loss_sum = nn.CrossEntropyLoss(reduction='sum')

    # Warm-up (Critical)
    for _ in range(20):
        ce_module.cross_entropy_bckwd_kernel(logits.data_ptr(), V, logsumexp.data_ptr(), input_ids.data_ptr(), grad_custom.data_ptr(), B)
        loss_pt = ce_loss_sum(logits, input_ids)
        loss_pt.backward()
        logits.grad.zero_()

    torch.cuda.synchronize()
    logits.grad.zero_()


    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("custom_ce_kernel"):
            ce_module.cross_entropy_bckwd_kernel(
                logits.data_ptr(),
                V,
                logsumexp.data_ptr(),
                input_ids.data_ptr(),
                grad_custom.data_ptr(),
                B
            )
        
        with record_function("pytorch_ce"):
                loss_pt = ce_loss_sum(logits, input_ids)
                loss_pt.backward()
                torch.cuda.synchronize()
        logits.grad.zero_()
            
        
        with record_function("unsloth_triton"):
            # sum loss to make it scalar
            loss_un = unsloth.kernels.cross_entropy_loss.Fast_CrossEntropyLoss.apply(logits, input_ids).sum()
            loss_un.backward()
            torch.cuda.synchronize()
        logits.grad.zero_()        

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    # chrome tracing
    prof.export_chrome_trace("ce_trace.json")


if __name__ == '__main__':
    if ce_module:
        bench_ce()