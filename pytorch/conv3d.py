import torch

DEPTHWISE = False

N = 1
IC = 64
D = 15
H = 75
W = 315
OC = 64
KD = 3
KH = 3
KW = 3

input_tensor = torch.randn(N, IC, D, H, W, dtype=torch.float16, device="cuda")
if not DEPTHWISE:
    input_tensor = input_tensor.contiguous(
        memory_format=torch.channels_last_3d)
conv3d = torch.nn.Conv3d(64, 64, kernel_size=3, padding=1,
                         bias=False, groups=IC if DEPTHWISE else 1).cuda().half()


@torch.compile
def compiled_conv3d(tensor):
    with torch.no_grad():
        return conv3d(tensor)


# warm up
output_tensor = compiled_conv3d(input_tensor)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output_tensor = compiled_conv3d(input_tensor)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end), "ms")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    output_tensor = compiled_conv3d(input_tensor)
print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=5))
