#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>   // HIP runtime
#include <nccl.h>              // RCCL provides NCCL-compatible API on ROCm

#define CHECK_HIP(call) do { \
  hipError_t _e = (call); \
  if (_e != hipSuccess) { \
    fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

#define CHECK_NCCL(call) do { \
  ncclResult_t _e = (call); \
  if (_e != ncclSuccess) { \
    fprintf(stderr, "NCCL/RCCL %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

inline void fill_with_index(float* dptr, int n, int gpu) {
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = gpu*1000.0f + i;
  CHECK_HIP(hipMemcpy(dptr, h.data(), n*sizeof(float), hipMemcpyHostToDevice));
}
__global__ void sgemm_naive(int M, int N, int K,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[row*K + k] * B[k*N + col];
    C[row*N + col] = acc;
  }
}

