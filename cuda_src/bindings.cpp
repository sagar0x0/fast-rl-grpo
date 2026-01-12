#include <stdint.h>
#include <torch/extension.h>

// Forward declarations of your launch functions in .cu
void cross_entropy_forward_launch(uintptr_t logits_ptr,
                                  int64_t logits_row_stride, uintptr_t loss_ptr,
                                  uintptr_t labels_ptr, uintptr_t lse_ptr,
                                  int rows);

void cross_entropy_backward_launch(uintptr_t logits_ptr,
                                   int64_t logits_row_stride,
                                   uintptr_t logsumexp_buf_ptr,
                                   uintptr_t labels_ptr,
                                   uintptr_t grad_logit_ptr,
                                   uintptr_t grad_output_ptr, int rows);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cross_entropy_forward_launch, "Cross Entropy Forward");
  m.def("backward", &cross_entropy_backward_launch, "Cross Entropy Backward");
}