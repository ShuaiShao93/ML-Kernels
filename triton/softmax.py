import triton
import triton.language as tl
import torch

num_rows, num_cols = 1208, 2460
x = torch.randn([num_rows, num_cols], device='cuda')
ref_y = torch.nn.Softmax(dim=1)(x)
y = torch.zeros_like(ref_y)


@triton.jit
def softmax(X, Y, input_row_stride, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = row_id * input_row_stride + col_offsets
    mask = col_offsets < input_row_stride
    x = tl.load(X+offsets, mask=mask, other=-float('inf'))

    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    tl.store(Y+offsets, y, mask)


BLOCK_SIZE = triton.next_power_of_2(num_cols)
grid = (num_rows,)
softmax[grid](x, y, num_cols, BLOCK_SIZE=BLOCK_SIZE)

assert torch.allclose(y, ref_y), (y, ref_y)
print("passed")
