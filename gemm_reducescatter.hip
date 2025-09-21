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
int main() {
  const int world = 8;
  int devs[world]; for (int i=0;i<world;++i) devs[i]=i;

  ncclComm_t comms[world];
  CHECK_NCCL(ncclCommInitAll(comms, world, devs));

  const int M = 1024, K = 512, N = 1024;
  const int rows_per_rank = M / world;

  hipStream_t streams[world];
  float *A[world], *B[world], *Cpartial[world], *Cstrip[world];

  dim3 block(16,16);
  dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);

  for (int r=0;r<world;++r) {
    CHECK_HIP(hipSetDevice(r));
    CHECK_HIP(hipStreamCreate(&streams[r]));
    CHECK_HIP(hipMalloc(&A[r], M*K*sizeof(float)));
    CHECK_HIP(hipMalloc(&B[r], K*N*sizeof(float)));
    CHECK_HIP(hipMalloc(&Cpartial[r], M*N*sizeof(float)));
    CHECK_HIP(hipMalloc(&Cstrip[r], rows_per_rank*N*sizeof(float)));
    fill_with_index(A[r], M*K, r);
    fill_with_index(B[r], K*N, r);
    hipLaunchKernelGGL(sgemm_naive, grid, block, 0, streams[r], M,N,K,A[r],B[r],Cpartial[r]);
  }

  for (int r=0;r<world;++r) {
    CHECK_HIP(hipSetDevice(r));
    CHECK_NCCL(ncclReduceScatter(Cpartial[r],
                                 Cstrip[r],
                                 rows_per_rank*N,
                                 ncclFloat,
                                 ncclSum,
                                 comms[r],
                                 streams[r]));
  }

  for (int r=0;r<world;++r) {
    CHECK_HIP(hipSetDevice(r));
    CHECK_HIP(hipStreamSynchronize(streams[r]));
    CHECK_HIP(hipFree(A[r]));
    CHECK_HIP(hipFree(B[r]));
    CHECK_HIP(hipFree(Cpartial[r]));
    CHECK_HIP(hipFree(Cstrip[r]));
    CHECK_HIP(hipStreamDestroy(streams[r]));
    ncclCommDestroy(comms[r]);
  }
  printf("GEMM + ReduceScatter (HIP/RCCL): OK\n");
  return 0;
}
