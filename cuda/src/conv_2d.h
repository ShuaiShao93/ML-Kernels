#pragma once

#include <cuda_runtime.h>

void conv2d(const float *A, const float *B, float *C, int N, int H, int W,
            int kH, int kW, int in_C, int out_C, cudaStream_t stream);