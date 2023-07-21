
__global__ void gemm_kernel(const float *a_ptr, const float *b_ptr,
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

void gemm(const float *a_ptr, const float *b_ptr, float *c_ptr, int M, int N,
          int K, int stride_am, int stride_ak, int stride_bk, int stride_bn,
          int stride_cm, int stride_cn, cudaStream_t stream) {
  dim3 blockDim(32, 32, 1);
  dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32, 1);
  gemm_kernel<<<gridDim, blockDim, 0, stream>>>(
      a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
      stride_cm, stride_cn);
}