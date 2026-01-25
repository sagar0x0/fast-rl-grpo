#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

// 2-d grid launch

__global__ void
cross_entropy_backward_kernel(const __nv_bfloat16 *__restrict__ logits,
                              const int64_t *__restrict__ labels,
                              const float *__restrict__ logsumexp_buff,
                              __nv_bfloat16 *__restrict__ grad_logits,
                              int64_t logits_row_stride,
                              const int v_per_block) {
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

            float grad_val1 = __expf(val1 - logsumexp) -
                                (target_label == global_idx1 ? 1.0f : 0.0f);
            float grad_val2 = __expf(val2 - logsumexp) -
                                (target_label == global_idx2 ? 1.0f : 0.0f);

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
            __expf(val - logsumexp) - (target_label == global_idx ? 1.0f : 0.0f);

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

    const int v_per_block = 2048; // 2048 vocab per block

    dim3 grid(rows, (logits_row_stride + v_per_block - 1) /
                        v_per_block); // 2048 vocab per block
    dim3 block(block_size);

    cross_entropy_backward_kernel<<<grid, block>>>(logits, labels, logsumexp_buf,
                                                    grad_logits, logits_row_stride,
                                                    v_per_block);

    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(cross_entropy_bckwd_kernel, m) {
    m.def("cross_entropy_bckwd_kernel", &cross_entropy_backward_launch,
            "Cuda cross_entropy backward kernel");
}