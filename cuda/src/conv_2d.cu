#include "conv_2d.h"

#include <iostream>

__global__ void conv2d_kernel(const float *A, const float *B, float *C, int N,
                              int H, int W, int kH, int kW, int in_C,
                              int out_C) {
  int out_c = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = y % W;
  int h = (y / W) % H;
  int n = (y / W) / H;

  int stride_an = H * W * in_C;
  int stride_ah = W * in_C;
  int stride_aw = in_C;
  int stride_aic = 1;

  int stride_boc = kH * kW * in_C;
  int stride_bkh = kW * in_C;
  int stride_bkw = in_C;
  int stride_bic = 1;

  int stride_cn = H * W * out_C;
  int stride_ch = W * out_C;
  int stride_cw = out_C;
  int stride_coc = 1;

  if (n >= N || h >= H || w >= W || out_c >= out_C) {
    return;
  }

  float accum = 0;
  for (int kh = 0; kh < kH; kh++) {
    for (int kw = 0; kw < kW; kw++) {
      for (int in_c = 0; in_c < in_C; in_c++) {
        float a;
        if (h + kh < 0 || h + kh >= H || w + kw < 0 || w + kw >= W) {
          a = 0;
        } else {
          a = A[n * stride_an + (h + kh) * stride_ah + (w + kw) * stride_aw +
                in_c * stride_aic];
        }
        float b = B[out_c * stride_boc + (kh - kW / 2) * stride_bkh +
                    (kw - kW / 2) * stride_bkw + in_c * stride_bic];
        accum += a * b;
      }
    }
  }

  C[n * stride_cn + h * stride_ch + w * stride_cw + out_c * stride_coc] = accum;
}

void conv2d(const float *A, const float *B, float *C, int N, int H, int W,
            int kH, int kW, int in_C, int out_C, cudaStream_t stream) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);

  constexpr int bm = 32, bn = 32;
  dim3 gridDim((out_C + bn - 1) / bn, (N * H * W + bm - 1) / bm);
  dim3 blockDim(bn, bm);
  conv2d_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, N, H, W, kH, kW,
                                                  in_C, out_C);

  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cout << "Elapsed time: " << time << std::endl;
}