#include "../src/conv_2d.h"

#include "utils.h"
#include <cassert>

int main(int argc, char **argv) {
  bool VERIFY = true;

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

  if (VERIFY) {
    float *ref_c = new float[N * H * W * out_C];
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

    for (int n = 0; n < N; n++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int oc = 0; oc < out_C; oc++) {
            float accum = 0;
            for (int kh = 0; kh < kH; kh++) {
              for (int kw = 0; kw < kW; kw++) {
                for (int ic = 0; ic < in_C; ic++) {
                  float a;
                  if (h - kH / 2 + kh < 0 || h - kH / 2 + kh >= H ||
                      w - kW / 2 + kw < 0 || w - kW / 2 + kw >= W) {
                    a = 1;
                  } else {
                    a = host_A[n * stride_an + (h - kH / 2 + kh) * stride_ah +
                               (w - kW / 2 + kw) * stride_aw + ic * stride_aic];
                  }
                  accum += a * host_B[oc * stride_boc + kh * stride_bkh +
                                      kw * stride_bkw + ic * stride_bic];
                }
              }
            }
            ref_c[n * stride_cn + h * stride_ch + w * stride_cw +
                  oc * stride_coc] = accum;
          }
        }
      }
    }

    float *host_c = new float[N * H * W * out_C];
    assert(cudaMemcpy(host_c, C, N * H * W * out_C * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    if (verify_matrix(host_c, ref_c, N * H * W * out_C)) {
      std::cout << "Verified and Passed" << std::endl;
    }
  } else {
    std::cout << "Passed" << std::endl;
  }
}