import tensorrt as trt
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

N = 1
IC = 64
D = 15
H = 75
W = 315
OC = 64
KD = 3
KH = 3
KW = 3

INPUT_SHAPE = (N, IC, D, H, W)
INPUT_SIZE = N * IC * D * H * W
OUTPUT_SHAPE = (N, OC, D, H, W)
OUTPUT_SIZE = N * OC * D * H * W
KERNEL_SHAPE = (OC, IC, KD, KH, KW)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

trt_logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(trt_logger)

network = builder.create_network(EXPLICIT_BATCH)
input = network.add_input(name="input", shape=INPUT_SHAPE, dtype=trt.float16)
input.allowed_formats = 1 << int(trt.TensorFormat.DHWC8)
conv3d = network.add_convolution_nd(
    input, OC, [KD, KH, KW], np.ones(KERNEL_SHAPE, dtype=np.float16))
conv3d.padding_nd = (1, 1, 1)
conv3d.stride_nd = (1, 1, 1)

# Useless with trt.BuilderFlag.FP16 below
# conv3d.precision = trt.float16

# Seems useless with "output.dtype = trt.DataType.HALF" below
# conv3d.set_output_type(0, trt.DataType.HALF)

output = conv3d.get_output(0)
output.name = "output"
output.dtype = trt.DataType.HALF # required
output.allowed_formats = 1 << int(trt.TensorFormat.DHWC8)
network.mark_output(output)
assert output.shape == OUTPUT_SHAPE

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.DIRECT_IO)
config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

engine = builder.build_engine(network, config)
inspector = engine.create_engine_inspector()
# Row major means channel first, channel major means channel last.
print('trt_engine layer_info:\n{}'.format(
    inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    ))

context = engine.create_execution_context()

names = [_ for _ in engine]
input_names = list(filter(engine.binding_is_input, names))
output_names = list(set(names) - set(input_names))
num_bindings = len(input_names) + len(output_names)
bindings = [None] * num_bindings

inputs = {}
for name in input_names:
    input_idx = engine.get_binding_index(name)
    inputs[name] = cuda.mem_alloc(INPUT_SIZE * 2)
    bindings[input_idx] = int(inputs[name])

outputs = {}
for name in output_names:
    output_idx = engine.get_binding_index(name)
    outputs[name] = cuda.mem_alloc(OUTPUT_SIZE * 2)
    bindings[output_idx] = int(outputs[name])

stream = cuda.Stream()
# warm up
context.execute_async_v2(bindings, stream.handle)

start, end = cuda.Event(), cuda.Event()
start.record()
context.execute_async_v2(bindings, stream.handle)
end.record()
end.synchronize()
ms = start.time_till(end)
print("Time Elapsed", ms)

stream.synchronize()

print("DONE")
