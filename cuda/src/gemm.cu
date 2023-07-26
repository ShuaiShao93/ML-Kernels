#include "gemm.h"

#include <iostream>

__global__ void gemm_kernel_0(const float *a_ptr, const float *b_ptr,
                              float *c_ptr, int M, int N, int K, int stride_am,
                              int stride_ak, int stride_bk, int stride_bn,
                              int stride_cm, int stride_cn) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= M || y >= N) {
    return;
  }

  float accum = 0;
  for (int k = 0; k < K; k++) {
    accum += a_ptr[x * stride_am + k * stride_ak] *
             b_ptr[k * stride_bk + y * stride_bn];
  }
  c_ptr[x * stride_cm + y * stride_cn] = accum;
}

__global__ void gemm_kernel_1(const float *a_ptr, const float *b_ptr,
                              float *c_ptr, int M, int N, int K, int stride_am,
                              int stride_ak, int stride_bk, int stride_bn,
                              int stride_cm, int stride_cn) {
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (m >= M || n >= N) {
    return;
  }

  float accum = 0;
  for (int k = 0; k < K; k++) {
    accum += a_ptr[m * stride_am + k * stride_ak] *
             b_ptr[k * stride_bk + n * stride_bn];
  }
  c_ptr[m * stride_cm + n * stride_cn] = accum;
}

void gemm(const float *a_ptr, const float *b_ptr, float *c_ptr, int M, int N,
          int K, int stride_am, int stride_ak, int stride_bk, int stride_bn,
          int stride_cm, int stride_cn, cudaStream_t stream, int kernel_id) {
  switch (kernel_id) {
  case 0: {
    std::cout << "Using kernel " << kernel_id << ": naiive" << std::endl;
    dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
    dim3 blockDim(32, 32);
    gemm_kernel_0<<<gridDim, blockDim, 0, stream>>>(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn);
    break;
  }
  case 1: {
    std::cout << "Using kernel " << kernel_id << ": memory_colescing"
              << std::endl;
    dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
    dim3 blockDim(32, 32);
    gemm_kernel_1<<<gridDim, blockDim>>>(a_ptr, b_ptr, c_ptr, M, N, K,
                                         stride_am, stride_ak, stride_bk,
                                         stride_bn, stride_cm, stride_cn);
    break;
  }

  default:
    std::cerr << "Invalid kernel id " << kernel_id;
  }
}