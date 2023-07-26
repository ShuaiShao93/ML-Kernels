#pragma once

#include <cuda_runtime.h>

void gemm(const float *a_ptr, const float *b_ptr, float *c_ptr, int M, int N,
          int K, int stride_am, int stride_ak, int stride_bk, int stride_bn,
          int stride_cm, int stride_cn, cudaStream_t stream, int kernel_id);