#include "gemm.h"

#include <cstdio>
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
  // threadIdx.x is consecutive and should read contiguous memory.
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

// Unfortunately blockDim is not constexpr.
template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K>
__global__ void gemm_kernel_2(const float *a_ptr, const float *b_ptr,
                              float *c_ptr, int M, int N, int K, int stride_am,
                              int stride_ak, int stride_bk, int stride_bn,
                              int stride_cm, int stride_cn) {
  // Block offsets.
  int block_m = blockIdx.y * blockDim.y, block_n = blockIdx.x * blockDim.x;
  int thread_m = threadIdx.y, thread_n = threadIdx.x;
  a_ptr += block_m * stride_am;
  b_ptr += block_n * stride_bn;
  c_ptr += block_m * stride_cm + block_n * stride_cn;

  __shared__ float as[BLOCK_SIZE_M][BLOCK_SIZE_K],
      bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
  float accu = 0;

  for (int block_k = 0; block_k < K; block_k += BLOCK_SIZE_K) {
    // Assumes BLOCK_SIZE_M and BLOCK_SIZE_N are not less than BLOCK_SIZE_K
    if (thread_n < BLOCK_SIZE_K) {
      as[thread_m][thread_n] =
          a_ptr[thread_m * stride_am + thread_n * stride_ak];
    }
    if (thread_m < BLOCK_SIZE_K) {
      bs[thread_m][thread_n] =
          b_ptr[thread_m * stride_bk + thread_n * stride_bn];
    }

    a_ptr += BLOCK_SIZE_K * stride_ak;
    b_ptr += BLOCK_SIZE_K * stride_bk;

    __syncthreads();

    // Very critical. In first branch, the loop can be unrolled.
    // We can also have an if condition in the loop, but it would be
    // worse because of the branch.
    if (block_k + BLOCK_SIZE_K < K) {
      for (int k = 0; k < BLOCK_SIZE_K; k++) {
        accu += as[thread_m][k] * bs[k][thread_n];
      }
    } else {
      for (int k = 0; k < K - block_k; k++) {
        accu += as[thread_m][k] * bs[k][thread_n];
      }
    }

    __syncthreads();
  }

  if (block_m + thread_m < M && block_n + thread_n < N) {
    c_ptr[thread_m * stride_cm + thread_n * stride_cn] = accu;
  }
}

template <int BM, int BN, int BK, int TM>
__global__ void gemm_kernel_3(const float *A, const float *B, float *C, int M,
                              int N, int K, int stride_am, int stride_ak,
                              int stride_bk, int stride_bn, int stride_cm,
                              int stride_cn) {
  int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
  int thread_m = threadIdx.y, thread_n = threadIdx.x;
  A += block_m * stride_am;
  B += block_n * stride_bn;
  C += block_m * stride_cm + block_n * stride_cn;

  __shared__ float As[BM][BK], Bs[BK][BN];
  float accum[TM] = {0.};
  // Assumes thread block size M/N are not less than BK
  for (int block_k = 0; block_k < K; block_k += BK) {
    for (int m = thread_m * TM; m < thread_m * TM + TM; m += 1) {
      if (thread_n < BK) {
        As[m][thread_n] = A[m * stride_am + thread_n * stride_ak];
      }
    }
    if (thread_m < BK) {
      Bs[thread_m][thread_n] = B[thread_m * stride_bk + thread_n * stride_bn];
    }

    __syncthreads();

    A += BK * stride_ak;
    B += BK * stride_bk;

    // Assumes K is multiple of BK.
    for (int k = 0; k < BK; k += 1) {
      float b_tmp = Bs[k][thread_n];
      for (int m = thread_m * TM; m < thread_m * TM + TM; m++) {
        accum[m - thread_m * TM] += As[m][k] * b_tmp;
      }
    }

    __syncthreads();
  }

  for (int m = thread_m * TM; m < thread_m * TM + TM; m++) {
    if (block_m + m < M && block_n + thread_n < N) {
      C[m * stride_cm + thread_n * stride_cn] = accum[m - thread_m * TM];
    }
  }
}

void gemm(const float *a_ptr, const float *b_ptr, float *c_ptr, int M, int N,
          int K, int stride_am, int stride_ak, int stride_bk, int stride_bn,
          int stride_cm, int stride_cn, cudaStream_t stream, int kernel_id) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);

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
    dim3 gridDim((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    dim3 blockDim(32, 32);
    gemm_kernel_1<<<gridDim, blockDim, 0, stream>>>(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn);
    break;
  }
  case 2: {
    constexpr int block_size = 32;
    std::cout << "Using kernel " << kernel_id << ": shared_memory" << std::endl;
    dim3 gridDim((N + block_size - 1) / block_size,
                 (M + block_size - 1) / block_size);
    dim3 blockDim(block_size, block_size);
    gemm_kernel_2<block_size, block_size, block_size>
        <<<gridDim, blockDim, 0, stream>>>(a_ptr, b_ptr, c_ptr, M, N, K,
                                           stride_am, stride_ak, stride_bk,
                                           stride_bn, stride_cm, stride_cn);
    break;
  }
  case 3: {
    constexpr int bm = 64, bn = 64, bk = 8, tm = 8;
    std::cout << "Using kernel " << kernel_id << ": Register 1D Cache" << std::endl;
    dim3 gridDim((N + bn - 1) / bn, (M + bm - 1) / bm);
    dim3 blockDim(bn, bm / tm);
    gemm_kernel_3<bm, bn, bk, tm><<<gridDim, blockDim>>>(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn);
    break;
  }

  default:
    std::cerr << "Invalid kernel id " << kernel_id;
  }

  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cout << "Elaspsed time: " << time << std::endl;
}