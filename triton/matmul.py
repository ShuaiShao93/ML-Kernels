import math
import torch
import triton
import triton.language as tl

M = 400
N = 500
K = 32
a = torch.rand(M, K, device="cuda")
b = torch.rand(K, N, device="cuda")
ref_c = torch.matmul(a, b)
c = torch.zeros_like(ref_c)
print("ref_c", ref_c)


@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, K)
    a_ptrs = a_ptr + offsets_am[:, None] * K + offsets_k[None, :]
    b_ptrs = b_ptr + offsets_k[:, None] * N + offsets_bn[None, :]
    a = tl.load(a_ptrs, mask=offsets_k[None, :] < K, other=0)
    b = tl.load(b_ptrs, mask=offsets_k[:, None] < K, other=0)
    c = tl.dot(a, b)

    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offsets_cm[:, None] * N + offsets_cn[None, :]
    c_mask = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
grid = (math.ceil(M / BLOCK_SIZE_M),  math.ceil(N / BLOCK_SIZE_N))
matmul[grid](a, b, c, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N)
print("c", c)

assert torch.allclose(ref_c, c)
print("Passed")
