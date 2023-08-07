#include "../src/conv_2d.h"

#include "utils.h"
#include <cassert>

int main(int argc, char **argv) {
  int N = 3, H = 200, W = 300, in_C = 32, kH = 3, kW = 3, out_C = 64;
  float *A, *B, *C;
  assert(cudaMalloc(&A, N * H * W * in_C * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&B, out_C * kH * kW * in_C * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&C, N * H * W * out_C * sizeof(float)) == cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *host_A = new float[N * H * W * in_C];
  float *host_B = new float[out_C * kH * kW * in_C];
  randomize_matrix(host_A, N * H * W * in_C);
  randomize_matrix(host_B, out_C * kH * kW * in_C);

  assert(cudaMemcpyAsync(A, host_A, N * H * W * in_C * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpyAsync(B, host_B, out_C * kH * kW * in_C * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  conv2d(A, B, C, N, H, W, kH, kW, in_C, out_C, stream);
  auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    std::cerr << "Kernel failed: " << status << std::endl;
    assert(status == cudaSuccess);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
}