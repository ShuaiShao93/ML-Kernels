import math
import torch
import triton
import triton.language as tl

M = 500
N = 400
K = 128
a = torch.rand(M, K, device="cuda")
b = torch.rand(K, N, device="cuda")
ref_c = torch.matmul(a, b)
c = torch.zeros_like(ref_c)
print("ref_c", ref_c)


@triton.jit
def matmul(a_ptr, b_ptr, c_ptr,
           M, N, K,
           stride_am, stride_ak,
           stride_bk, stride_bn,
           stride_cm, stride_cn,
           BLOCK_SIZE_M: tl.constexpr,
           BLOCK_SIZE_N: tl.constexpr,
           BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    am_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bn_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + am_offsets[:, None] * \
            stride_am + k_offsets[None, :] * stride_ak
        a_mask = (am_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b_ptrs = b_ptr + k_offsets[:, None] * \
            stride_bk + bn_offsets[None, :] * stride_bn
        b_mask = (k_offsets[:, None] < K) & (bn_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        accum += tl.dot(a, b)

    c_ptrs = c_ptr + am_offsets[:, None] * \
        stride_cm + bn_offsets[None, :] * stride_cn
    c_mask = (am_offsets[:, None] < M) & (bn_offsets[None, :] < N)
    tl.store(c_ptrs, accum, mask=c_mask)


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
grid = (math.ceil(M / BLOCK_SIZE_M),  math.ceil(N / BLOCK_SIZE_N))
stride_am, stride_ak = K, 1
stride_bk, stride_bn = N, 1
stride_cm, stride_cn = N, 1
matmul[grid](a, b, c, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
             stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages=2)
print("c", c)

assert torch.allclose(ref_c, c)
print("Passed")
