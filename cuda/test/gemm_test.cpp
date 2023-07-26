#include "../src/gemm.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  // FLOPS = 2*4092^3 + 4092^2 = 137 GFLOPS
  int M = 4000, N = 4000, K = 4092;
  float *a_ptr, *b_ptr, *c_ptr;
  assert(cudaMalloc(&a_ptr, M * K * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&b_ptr, K * N * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&c_ptr, M * N * sizeof(float)) == cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> host_a(M * K, 1);
  std::vector<float> host_b(K * N, 1);
  assert(cudaMemcpyAsync(a_ptr, host_a.data(), M * K * sizeof(float),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
  assert(cudaMemcpyAsync(b_ptr, host_b.data(), K * N * sizeof(float),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);

  int kernel_id = 0;
  gemm(a_ptr, b_ptr, c_ptr, M, N, K, K, 1, N, 1, N, 1, stream, kernel_id);

  std::vector<float> host_c(M * N);
  assert(cudaMemcpyAsync(host_c.data(), c_ptr, M * N * sizeof(float),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(host_c[0] == 4092);

  std::cout << "Passed" << std::endl;
}