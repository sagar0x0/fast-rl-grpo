#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// forward kernel cross entropy

__device__ __forceinline__ void warp_reduce_online(float &m, float &d) {
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

__global__ void cross_entropy_forward_kernel(
    const __nv_bfloat16 *__restrict__ logits,
    const int64_t *__restrict__ labels, __nv_bfloat16 *__restrict__ loss,
    float *__restrict__ lse, int64_t logits_row_stride) {
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

  // vectorized load
  const float4 *vec_logits = reinterpret_cast<const float4 *>(row_logits);

  int num_vectors = logits_row_stride / 8;

  // local register accumulation (colaesced memory loads)
  for (int i = tid; i < num_vectors; i += blockDim.x) {
    float4 vec_bits = vec_logits[i];
    // vec contains 8 bloat16 or 4 bfloat16 pairs of raw bits

    // __nv_bfloat162 cast it to seperate pair of __nv_bfloat16
    __nv_bfloat162 *pairs = reinterpret_cast<__nv_bfloat162 *>(&vec_bits);

#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 val = __bfloat1622float2(pairs[j]);

      float m_prev = m;
      m = fmaxf(m_prev, fmaxf(val.x, val.y));

      d = d * (__expf(m_prev - m)) + __expf(val.x - m) + __expf(val.y - m);
    }
  }

  // leftover
  int tail_start = num_vectors * 8;

  // Every thread checks if there are leftovers it needs to handle
  for (int i = tail_start + tid; i < logits_row_stride; i += blockDim.x) {
    // Load explicitly as bfloat16
    __nv_bfloat16 val_bf16 = row_logits[i];
    float val = __bfloat162float(val_bf16);

    float m_prev = m;
    m = fmaxf(m_prev, val);
    d = d * (__expf(m_prev - m)) + __expf(val - m);
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

    warp_reduce_online(m, d);
  }

  // final loss calculation

  if (tid == 0) {
    float log_sum_exp = logf(d);
    float target_logit = __bfloat162float(row_logits[label]);
    loss[batch_idx] = __float2bfloat16(log_sum_exp - target_logit + m);
    lse[batch_idx] = log_sum_exp;
  }
}

void cross_entropy_forward_launch(uintptr_t logits_ptr,      // logits [B*L, V]
                                  int64_t logits_row_stride, // V
                                  uintptr_t loss_ptr,        // [B*L]
                                  uintptr_t labels_ptr,      // [B*L]
                                  uintptr_t lse_ptr,         // [B*L]
                                  int rows                   // B*L
) {
  auto *logits = reinterpret_cast<__nv_bfloat16 *>(logits_ptr);
  auto *loss = reinterpret_cast<__nv_bfloat16 *>(loss_ptr);
  auto *labels = reinterpret_cast<int64_t *>(labels_ptr);
  auto *lse = reinterpret_cast<float *>(lse_ptr);

  // Optimal configuration - tune based on your hardware
  const int block_size = 256; // 8 warps
  const int num_warps = block_size / 32;
  const size_t shared_mem_size =
      2 * num_warps * sizeof(float); // float is acc scalar type

  dim3 grid(rows);
  dim3 block(block_size);

  cross_entropy_forward_kernel<<<grid, block, shared_mem_size>>>(
      logits, labels, loss, lse, logits_row_stride);

  // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// backward kernel cross entropy

__global__ void cross_entropy_backward_kernel(
    const __nv_bfloat16 *__restrict__ logits,
    const int64_t *__restrict__ labels,
    const float *__restrict__ logsumexp_buff,
    __nv_bfloat16 *__restrict__ grad_logits, int64_t logits_row_stride,
    const __nv_bfloat16 *__restrict__ grad_output, const int v_per_block) {
  const int batch_idx = blockIdx.x;
  const int vocab_idx = blockIdx.y; // tile idx
  const int tid = threadIdx.x;
  const int lane_id = tid / 32;
  const int warp_id = tid % 32;

  // start and range of curr tile
  int start_col = vocab_idx * v_per_block;
  // last block safe memory access (eg. 352 ele foe last block of 156000 vocab
  // given 2048 v_per_block)
  int num_ele_vocab = min((int)v_per_block, (int)logits_row_stride - start_col);

  // Load the specific weight for this row (Advantage * Mask / N)
  float dloss = __bfloat162float(grad_output[batch_idx]);
  // if (labels[batch_idx] == -100) dloss = 0.0f;   comp_mask handles in grpo

  const __nv_bfloat16 *row_logits =
      logits + batch_idx * logits_row_stride + start_col;
  __nv_bfloat16 *row_grads =
      grad_logits + batch_idx * logits_row_stride + start_col;

  // logsumexp for this row
  float logsumexp = logsumexp_buff[batch_idx];
  // target input_ids yi : for target == 1, for non-target == 0
  int64_t target_label = labels[batch_idx];

  // vectorized loads
  const float4 *vec_logits = reinterpret_cast<const float4 *>(row_logits);
  float4 *vec_grad = reinterpret_cast<float4 *>(row_grads);

  // vec idx
  int num_vec_ele = num_ele_vocab / 8;

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
      // we need global idx for target label
      int global_idx1 = start_col + (i * 8 + j * 2);
      int global_idx2 = start_col + (i * 8 + j * 2 + 1);

      float grad_val1 = (__expf(val1 - logsumexp) -
                         (target_label == global_idx1 ? 1.0f : 0.0f)) *
                        dloss;
      float grad_val2 = (__expf(val2 - logsumexp) -
                         (target_label == global_idx2 ? 1.0f : 0.0f)) *
                        dloss;

      grad_pairs[j] = __floats2bfloat162_rn(grad_val1, grad_val2);
    }

    // store all 8 elements
    vec_grad[i] = out_grad_vec;
  }

  // leftover tali processing
  int tail_start = num_vec_ele * 8;
  // Every thread checks if there are leftovers it needs to handle
  for (int i = tail_start + tid; i < num_ele_vocab; i += blockDim.x) {
    // Load explicitly as bfloat16
    int global_idx = start_col + i;
    float val = __bfloat162float(row_logits[i]);
    float grad_val =
        (__expf(val - logsumexp) - (target_label == global_idx ? 1.0f : 0.0f)) *
        dloss;

    row_grads[i] = __float2bfloat16(grad_val);
  }
}

void cross_entropy_backward_launch(uintptr_t logits_ptr,      // logits [B*L, V]
                                   int64_t logits_row_stride, // V
                                   uintptr_t logsumexp_buf_ptr, // [B*L]
                                   uintptr_t labels_ptr,        // [B*L]
                                   uintptr_t grad_logit_ptr,    // [B*L, V]
                                   uintptr_t grad_output_ptr,   // [B*L]
                                   int rows                     // B*L
) {
  auto *logits = reinterpret_cast<__nv_bfloat16 *>(logits_ptr);
  auto *logsumexp_buf = reinterpret_cast<float *>(logsumexp_buf_ptr);
  auto *labels = reinterpret_cast<int64_t *>(labels_ptr);
  auto *grad_logits = reinterpret_cast<__nv_bfloat16 *>(grad_logit_ptr);
  auto *grad_output = reinterpret_cast<__nv_bfloat16 *>(grad_output_ptr);
  // Optimal configuration - tune based on your hardware
  const int block_size = 256; // 8 warps
  // const int num_warps = block_size / 32;
  // shared memory not used as there is no reduction of threads

  const int v_per_block = 2048; // 2048 vocab per block

  dim3 grid(rows, (logits_row_stride + v_per_block - 1) /
                      v_per_block); // 2048 vocab per block
  dim3 block(block_size);

  cross_entropy_backward_kernel<<<grid, block>>>(logits, labels, logsumexp_buf,
                                                 grad_logits, logits_row_stride,
                                                 grad_output, v_per_block);

  // C10_CUDA_KERNEL_LAUNCH_CHECK();
}
