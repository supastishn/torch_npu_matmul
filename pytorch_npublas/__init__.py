import torch
import _pytorch_npublas
def matmul_patch(input_a, input_b, *, out=None, backend_choice=1):
    if input_a.is_cuda or input_b.is_cuda:
        return torch._matmul_original(input_a, input_b, out=out)
    if input_a.dtype != torch.float32 or input_b.dtype != torch.float32:
        return torch._matmul_original(input_a, input_b, out=out)
    dim_a = input_a.dim()
    dim_b = input_b.dim()
    if dim_a == 1 and dim_b == 2:
        res = matmul_patch(input_a.unsqueeze(0), input_b, backend_choice=backend_choice)
        res = res.squeeze(0)
        if out is not None:
            out.copy_(res)
            return out
        return res
    if dim_a == 2 and dim_b == 1:
        res = matmul_patch(input_a, input_b.unsqueeze(1), backend_choice=backend_choice)
        res = res.squeeze(1)
        if out is not None:
            out.copy_(res)
            return out
        return res
    if dim_a == 2 and dim_b == 2:
        a_contig = input_a.contiguous()
        b_contig = input_b.contiguous()
        if out is not None:
            out_contig = out.contiguous()
            _pytorch_npublas.matmul_out(a_contig, b_contig, out_contig, backend_choice)
            if not out.is_contiguous():
                out.copy_(out_contig)
            return out
        return _pytorch_npublas.matmul(a_contig, b_contig, backend_choice)
    if dim_a > 2 and dim_b == 2:
        orig_shape = list(input_a.shape)
        orig_shape[-1] = input_b.shape[-1]
        a_reshaped = input_a.reshape(-1, input_a.shape[-1])
        res = matmul_patch(a_reshaped, input_b, backend_choice=backend_choice)
        res = res.reshape(orig_shape)
        if out is not None:
            out.copy_(res)
            return out
        return res
    try:
        batch_shape_a = input_a.shape[:-2]
        batch_shape_b = input_b.shape[:-2]
        dummy_a = torch.empty(batch_shape_a)
        dummy_b = torch.empty(batch_shape_b)
        broadcast_batch = torch.broadcast_shapes(batch_shape_a, batch_shape_b)
        a_expanded = input_a.expand(broadcast_batch + input_a.shape[-2:])
        b_expanded = input_b.expand(broadcast_batch + input_b.shape[-2:])
        a_flat = a_expanded.reshape(-1, input_a.shape[-2], input_a.shape[-1])
        b_flat = b_expanded.reshape(-1, input_b.shape[-2], input_b.shape[-1])
        results = []
        for i in range(a_flat.shape[0]):
            res_slice = matmul_patch(a_flat[i], b_flat[i], backend_choice=backend_choice)
            results.append(res_slice)
        res = torch.stack(results).reshape(broadcast_batch + (input_a.shape[-2], input_b.shape[-1]))
        if out is not None:
            out.copy_(res)
            return out
        return res
    except Exception:
        return torch._matmul_original(input_a, input_b, out=out)
if not hasattr(torch, "_matmul_original"):
    torch._matmul_original = torch.matmul
    torch.matmul = matmul_patch