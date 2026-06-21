import torch
import _pytorch_npublas
def matmul_patch(input_a, input_b, *, out=None):
    if input_a.is_cuda or input_b.is_cuda:
        return torch._matmul_original(input_a, input_b, out=out)
    if input_a.dim() == 2 and input_b.dim() == 2 and input_a.dtype == torch.float32 and input_b.dtype == torch.float32:
        if out is not None:
            _pytorch_npublas.matmul_out(input_a, input_b, out)
            return out
        return _pytorch_npublas.matmul(input_a, input_b)
    return torch._matmul_original(input_a, input_b, out=out)
if not hasattr(torch, "_matmul_original"):
    torch._matmul_original = torch.matmul
    torch.matmul = matmul_patch