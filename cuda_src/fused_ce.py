import torch
import fused_ce_cuda # compile  module (globaly available)


@torch.library.custom_op("custom_ce::fused_ce_fwd", mutates_args=())
def fused_ce_fwd(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # secure memory layout
    logits = logits.contiguous() 
    labels = labels.contiguous()

    # logits: [B * L, V],   labels: [B * L]
    num_rows, row_stride  = logits.shape  # B*L, V

    loss = torch.empty([num_rows], dtype=logits.dtype, device=logits.device) # dtype: bfloat16, device: cuda
    lse = torch.empty([num_rows], dtype=torch.float32, device=logits.device) # dtype: float32, device: cuda

    # call Custom Forward Kernel
    fused_ce_cuda.forward(
        logits.data_ptr(),
        row_stride,
        loss.data_ptr(),
        labels.data_ptr(),
        lse.data_ptr(),
        num_rows
    )
    
    return loss, lse

# fake meta implementation
# tells torch.compile what the output shapes/types
@fused_ce_fwd.register_fake
def _(logits, labels):
    num_rows, _ = logits.shape
    
    loss = torch.empty([num_rows,], dtype=logits.dtype, device=logits.device)
    lse = torch.empty([num_rows,], dtype=torch.float32, device=logits.device)
    return loss, lse



@torch.library.custom_op("custom_ce::fused_ce_bwd", mutates_args=())
def fused_ce_bwd(grad_output: torch.Tensor, logits: torch.Tensor, 
                 lse: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # secure memory layout
    grad_output = grad_output.contiguous()
    logits = logits.contiguous()
    lse = lse.contiguous()
    labels = labels.contiguous()

    num_rows, row_stride = logits.shape
    grad_logits = torch.empty_like(logits)

    fused_ce_cuda.backward(
        logits.data_ptr(),
        row_stride,
        lse.data_ptr(),
        labels.data_ptr(),
        grad_logits.data_ptr(),
        grad_output.data_ptr(),
        num_rows
    )
    
    return grad_logits


@fused_ce_bwd.register_fake
def _(grad_output, logits, lse, labels):
    # same shape as logits
    return torch.empty_like(logits)


# connect with autograd
class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        # call registed custom op
        loss, lse = torch.ops.custom_ce.fused_ce_fwd(logits, labels)
         
        # save the ctx values for backward pass
        ctx.save_for_backward(logits, labels, lse)
        
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        # grad_output is the gradient of the final scalar loss w.r.t. our forward 'loss' output
        # load the saved ctx
        logits, labels, lse = ctx.saved_tensors
        # grad_output needs to be contiguous
        grad_output = grad_outputs[0]

        grad_logits = torch.ops.custom_ce.fused_ce_bwd(grad_output, logits, lse, labels)
        
        return grad_logits, None