#include "../src/gemm.h"
#include "utils.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char **argv) {
  bool VERIFY = false;
  // FLOPS = 2*4092^3 + 4092^2 = 137 GFLOPS
  int M = 4000, N = 4000, K = 4092;

  if (VERIFY) {
    M = 500, N = 400, K = 128;
  }
  float *a_ptr, *b_ptr, *c_ptr, *ref_c_ptr;
  assert(cudaMalloc(&a_ptr, M * K * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&b_ptr, K * N * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&c_ptr, M * N * sizeof(float)) == cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *host_a = new float[M * K];
  float *host_b = new float[K * N];
  randomize_matrix(host_a, M * K);
  randomize_matrix(host_b, K * N);

  float *host_ref_c;
  if (VERIFY) {
    host_ref_c = new float[M * N];
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        float accu = 0;
        for (int k = 0; k < K; k++) {
          accu += host_a[m * K + k] * host_b[k * N + n];
        }
        host_ref_c[m * N + n] = accu;
      }
    }
  }

  assert(cudaMemcpyAsync(a_ptr, host_a, M * K * sizeof(float),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
  assert(cudaMemcpyAsync(b_ptr, host_b, K * N * sizeof(float),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);

  int kernel_id = 4;
  gemm(a_ptr, b_ptr, c_ptr, M, N, K, K, 1, N, 1, N, 1, stream, kernel_id);

  auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    std::cerr << "Kernel failed: " << status << std::endl;
    assert(status == cudaSuccess);
  }
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    std::cerr << "Kernel failed: " << status << std::endl;
    assert(status == cudaSuccess);
  }

  float *host_c = new float[M * N];
  assert(cudaMemcpyAsync(host_c, c_ptr, M * N * sizeof(float),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  if (VERIFY) {
    if (verify_matrix(host_c, host_ref_c, M * N)) {
      std::cout << "Verified and Passed" << std::endl;
    } else {
      return 1;
    }
  } else {
    std::cout << "Passed" << std::endl;
  }
}