#include "utils.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define CHECK_CUDNN(expression)                                                \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

void CudnnConv3D(const float *A, const float *B, float *C, int N, int D, int H,
                 int W, int kD, int kH, int kW, int in_C, int out_C,
                 float alpha, float beta, cudaStream_t stream) {
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t A_desc, C_desc;
  cudnnFilterDescriptor_t B_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t algo =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  void *workspace;
  size_t workspace_size;

  cudnnCreate(&handle);
  CHECK_CUDNN(cudnnSetStream(handle, stream));
  cudnnCreateTensorDescriptor(&A_desc);
  cudnnCreateTensorDescriptor(&C_desc);
  cudnnCreateFilterDescriptor(&B_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  int dims_A[5] = {N, in_C, D, H, W};
  int strides_A[5] = {D * H * W * in_C, H * W * in_C, W * in_C, in_C, 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(A_desc, CUDNN_TENSOR_NHWC,
                                           CUDNN_DATA_FLOAT, 5, dims_A));

  int dims_B[5] = {out_C, in_C, kD, kH, kW};
  CHECK_CUDNN(cudnnSetFilterNdDescriptor(B_desc, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NHWC, 5, dims_B));

  int dims_C[5] = {N, out_C, D, H, W};
  CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(C_desc, CUDNN_TENSOR_NHWC,
                                           CUDNN_DATA_FLOAT, 5, dims_C));

  int pads[3] = {1, 1, 1};
  int strides[3] = {1, 1, 1};
  int dilations[3] = {1, 1, 1};
  CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
      conv_desc, 3, pads, strides, dilations, CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT));
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      handle, A_desc, B_desc, conv_desc, C_desc, algo, &workspace_size));
  if (workspace_size > 0) {
    assert(cudaMalloc(&workspace, workspace_size) == cudaSuccess);
  }

  // Warm up.
  CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, A_desc, A, B_desc, B,
                                      conv_desc, algo, workspace,
                                      workspace_size, &beta, C_desc, C));
  auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    std::cerr << "Kernel failed: " << status << std::endl;
    assert(status == cudaSuccess);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);

  CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, A_desc, A, B_desc, B,
                                      conv_desc, algo, workspace,
                                      workspace_size, &beta, C_desc, C));

  cudaEventRecord(end, stream);
  assert(cudaEventSynchronize(end) == cudaSuccess);
  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cout << "Elapsed time: " << time << " ms" << std::endl;

  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  cudnnDestroyTensorDescriptor(A_desc);
  cudnnDestroyTensorDescriptor(C_desc);
  cudnnDestroyFilterDescriptor(B_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
  cudaFree(workspace);
}

int main(int argc, char **argv) {
  bool VERIFY = true;

  int N = 2, D = 20, H = 10, W = 30, in_C = 32, kD = 3, kH = 3, kW = 3,
      out_C = 64;
  int size_A = N * D * H * W * in_C, size_B = out_C * kD * kH * kW * in_C,
      size_C = N * D * H * W * out_C;
  float *A, *B, *C;
  assert(cudaMalloc(&A, size_A * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&B, size_B * sizeof(float)) == cudaSuccess);
  assert(cudaMalloc(&C, size_C * sizeof(float)) == cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *host_A = new float[size_A];
  float *host_B = new float[size_B];
  randomize_matrix(host_A, size_A);
  randomize_matrix(host_B, size_B);

  assert(cudaMemcpyAsync(A, host_A, size_A * sizeof(float),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
  assert(cudaMemcpyAsync(B, host_B, size_B * sizeof(float),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess);

  float alpha = 3.2, beta = 0;
  CudnnConv3D(A, B, C, N, D, H, W, kD, kH, kW, in_C, out_C, alpha, beta,
              stream);

  if (VERIFY) {
    float *ref_c = new float[size_C];
    int stride_an = D * H * W * in_C;
    int stride_ad = H * W * in_C;
    int stride_ah = W * in_C;
    int stride_aw = in_C;
    int stride_aic = 1;

    int stride_boc = kD * kH * kW * in_C;
    int stride_bkd = kH * kW * in_C;
    int stride_bkh = kW * in_C;
    int stride_bkw = in_C;
    int stride_bic = 1;

    int stride_cn = D * H * W * out_C;
    int stride_cd = H * W * out_C;
    int stride_ch = W * out_C;
    int stride_cw = out_C;
    int stride_coc = 1;

    for (int n = 0; n < N; n++) {
      for (int d = 0; d < D; d++) {
        for (int h = 0; h < H; h++) {
          for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < out_C; oc++) {
              float accum = 0;
              for (int kd = 0; kd < kD; kd++) {
                for (int kh = 0; kh < kH; kh++) {
                  for (int kw = 0; kw < kW; kw++) {
                    for (int ic = 0; ic < in_C; ic++) {
                      float a;
                      if (d - kD / 2 + kd < 0 || d - kD / 2 + kd >= D ||
                          h - kH / 2 + kh < 0 || h - kH / 2 + kh >= H ||
                          w - kW / 2 + kw < 0 || w - kW / 2 + kw >= W) {
                        a = 0;
                      } else {
                        a = host_A[n * stride_an +
                                   (d - kD / 2 + kd) * stride_ad +
                                   (h - kH / 2 + kh) * stride_ah +
                                   (w - kW / 2 + kw) * stride_aw +
                                   ic * stride_aic];
                      }
                      accum += a * host_B[oc * stride_boc + kd * stride_bkd +
                                          kh * stride_bkh + kw * stride_bkw +
                                          ic * stride_bic];
                    }
                  }
                }
              }
              ref_c[n * stride_cn + d * stride_cd + h * stride_ch +
                    w * stride_cw + oc * stride_coc] = accum * alpha;
            }
          }
        }
      }
    }

    float *host_c = new float[size_C];
    assert(cudaMemcpy(host_c, C, size_C * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    if (verify_matrix(host_c, ref_c, size_C)) {
      std::cout << "Verified and Passed" << std::endl;
    }
  } else {
    std::cout << "Passed" << std::endl;
  }
}