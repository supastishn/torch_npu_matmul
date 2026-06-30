import torch
import os
import _pytorch_npublas

if os.environ.get("NPUBLAS_VERBOSE") == "1":
    print(f"[PYTORCH_NPUBLAS] Loaded python package from: {__file__}")
    print(f"[PYTORCH_NPUBLAS] Loaded binary extension from: {_pytorch_npublas.__file__}")

_default_backend_choice = 1
_global_m_split = None
_global_k_split = None
_global_n_split = None

def set_backend(backend_choice):
    global _default_backend_choice
    _default_backend_choice = backend_choice
    _pytorch_npublas.set_backend(backend_choice)

def set_splits(m_split, k_split, n_split):
    global _global_m_split, _global_k_split, _global_n_split
    _global_m_split = m_split
    _global_k_split = k_split
    _global_n_split = n_split

class NPUMatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_a, input_b, backend_choice):
        ctx.save_for_backward(input_a, input_b)
        ctx.backend_choice = backend_choice
        m, k = input_a.shape
        n = input_b.shape[1]
        k_limit = int(os.environ.get("K_LIMIT", "0"))
        if (k_limit > 0 and k > k_limit) or m > 1024 or n > 1024:
            m_split = _global_m_split if _global_m_split is not None else max(1, (m + 1023) // 1024)
            k_split = _global_k_split if _global_k_split is not None else (max(1, (k + k_limit - 1) // k_limit) if k_limit > 0 else 1)
            n_split = _global_n_split if _global_n_split is not None else max(1, (n + 1023) // 1024)
            if os.environ.get("NPUBLAS_VERBOSE") == "1":
                print(f"[PYTORCH_NPUBLAS] MatMul Tiled Splits: m_split={m_split}, k_split={k_split}, n_split={n_split} (M_block={m // m_split}, K_block={k // k_split}, N_block={n // n_split})")
            return _pytorch_npublas.matmul_tiled(input_a, input_b, m_split, k_split, n_split, backend_choice)
        return _pytorch_npublas.matmul(input_a, input_b, backend_choice)

    @staticmethod
    def backward(ctx, grad_output):
        input_a, input_b = ctx.saved_tensors
        backend_choice = ctx.backend_choice
        grad_a = matmul_patch(grad_output, input_b.t(), backend_choice=backend_choice)
        grad_b = matmul_patch(input_a.t(), grad_output, backend_choice=backend_choice)
        return grad_a, grad_b, None

def matmul_patch(input_a, input_b, *, out=None, backend_choice=None):
    if backend_choice is None:
        backend_choice = _default_backend_choice
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
    if dim_a == 2 and dim_b == 2:
        m, k = input_a.shape
        n = input_b.shape[1]
        m_pad = ((m + 3) // 4) * 4
        k_pad = ((k + 3) // 4) * 4
        n_pad = ((n + 3) // 4) * 4
        if m_pad != m or k_pad != k or n_pad != n:
            a_padded = torch.zeros(m_pad, k_pad, dtype=input_a.dtype, device=input_a.device)
            a_padded[:m, :k] = input_a
            b_padded = torch.zeros(k_pad, n_pad, dtype=input_b.dtype, device=input_b.device)
            b_padded[:k, :n] = input_b
            res_padded = matmul_patch(a_padded, b_padded, backend_choice=backend_choice)
            res = res_padded[:m, :n]
            if out is not None:
                out.copy_(res)
                return out
            return res
        res = NPUMatMulFunction.apply(input_a.contiguous(), input_b.contiguous(), backend_choice)
        if out is not None:
            out.copy_(res)
            return out
        return res
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
import torch.nn.functional as F

def linear_patch(input, weight, bias=None):
    output = matmul_patch(input, weight.t())
    if bias is not None:
        output += bias
    return output

if not hasattr(F, "_linear_original"):
    F._linear_original = F.linear
    F.linear = linear_patch

if not hasattr(torch, "_mm_original"):
    torch._mm_original = torch.mm
    torch.mm = lambda input, mat2, *, out=None: matmul_patch(input, mat2, out=out)

def addmm_patch(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    mul_res = matmul_patch(mat1, mat2)
    res = beta * input + alpha * mul_res
    if out is not None:
        out.copy_(res)
        return out
    return res

if not hasattr(torch, "_addmm_original"):
    torch._addmm_original = torch.addmm
    torch.addmm = addmm_patch

if not hasattr(torch, "_matmul_original"):
    torch._matmul_original = torch.matmul
    torch.matmul = matmul_patch

def set_power_mode(power_mode):
    _pytorch_npublas.set_power_mode(power_mode)

def prepare_matmul(m, k, n, input_type=8, output_type=16):
    backend_choice = _default_backend_choice
    m = ((m + 3) // 4) * 4
    k = ((k + 3) // 4) * 4
    n = ((n + 3) // 4) * 4
    k_limit = int(os.environ.get("K_LIMIT", "0"))
    if (k_limit > 0 and k > k_limit) or m > 1024 or n > 1024:
        m_split = _global_m_split if _global_m_split is not None else max(1, (m + 1023) // 1024)
        k_split = _global_k_split if _global_k_split is not None else (max(1, (k + k_limit - 1) // k_limit) if k_limit > 0 else 1)
        n_split = _global_n_split if _global_n_split is not None else max(1, (n + 1023) // 1024)
        m_block = m // m_split
        k_block = k // k_split
        n_block = n // n_split
        if _pytorch_npublas.has_graph(m_block, k_block, n_block, 8, 16):
            return
        _pytorch_npublas.prepare_matmul(m_block, k_block, n_block, 8, 16)
    else:
        if _pytorch_npublas.has_graph(m, k, n, 8, 16):
            return
        _pytorch_npublas.prepare_matmul(m, k, n, 8, 16)

try:
    _pytorch_npublas.initialize(_default_backend_choice)
except Exception:
    pass