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
		float log_sum_exp = logf(d) + m;
		float target_logit = __bfloat162float(row_logits[label]);
		loss[batch_idx] = __float2bfloat16(log_sum_exp - target_logit);
	}
}

void cross_entropy_forward_launch(uintptr_t logits_ptr, // logits [B*L, V]
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

PYBIND11_MODULE(cross_entropy_fwd_kernel, m) {
  m.def("cross_entropy_fwd_kernel", &cross_entropy_forward_launch,
        "Cuda cross_entropy kernel");
}