import triton
import triton.language as tl
import torch
import math

BLOCK_SIZE = 128
N = 1000
x = torch.rand([N], device='cuda')
y = torch.rand([N], device='cuda')
ref_z = x + y
z = torch.zeros_like(ref_z)


@triton.jit
def add(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask)
    y = tl.load(Y + offsets, mask)
    z = x + y
    tl.store(Z + offsets, z, mask)


grid = (math.ceil(N/BLOCK_SIZE),)
add[grid](x, y, z, N, BLOCK_SIZE)

assert torch.allclose(z, ref_z)
print("passed")
