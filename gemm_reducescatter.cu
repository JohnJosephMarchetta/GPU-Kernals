#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) do {                                 \
  cudaError_t _e = (call);                                     \
  if (_e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,    \
            cudaGetErrorString(_e));                           \
    std::exit(1);                                              \
  }                                                            \
} while(0)

#define CHECK_NCCL(call) do {                                  \
  ncclResult_t _e = (call);                                     \
  if (_e != ncclSuccess) {                                     \
    fprintf(stderr, "NCCL %s:%d: %s\n", __FILE__, __LINE__,    \
            ncclGetErrorString(_e));                           \
    std::exit(1);                                              \
  }                                                            \
} while(0)

inline void fill_with_index(float* p, int n, int gpu) {
  // quick host fill (debug); normally you'd cudaMemsetAsync or a kernel
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = gpu*1000.0f + i;
  CHECK_CUDA(cudaMemcpy(p, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
}

#pragma once

__global__ void sgemm_naive(int M, int N, int K,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
  int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..N)
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[row*K + k] * B[k*N + col];
    }
    C[row*N + col] = acc;
  }
}

// gemm_reducescatter.cu
#include <vector>
#include "helpers.h"
#include "sgemm_naive.cuh"

int main() {
  const int world = 8;
  int devs[world]; for (int i=0;i<world;++i) devs[i]=i;
  ncclComm_t comms[world];
  CHECK_NCCL(ncclCommInitAll(comms, world, devs));

  // problem sizes (tiny for demo)
  const int M = 1024, K = 512, N = 1024;
  const int rows_per_rank = M / world;     // assume divisible

  cudaStream_t streams[world];
  float *A[world], *B[world], *Cpartial[world], *Cstrip[world];

  dim3 block(16,16);
  dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);

  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamCreate(&streams[r]));
    CHECK_CUDA(cudaMalloc(&A[r], M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B[r], K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Cpartial[r], M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Cstrip[r], rows_per_rank*N*sizeof(float)));
    fill_with_index(A[r], M*K, r);
    fill_with_index(B[r], K*N, r);
    // local GEMM (naive)
    sgemm_naive<<<grid, block, 0, streams[r]>>>(M,N,K,A[r],B[r],Cpartial[r]);
  }

  // Each rank reduces (sum) across all ranks and scatters a contiguous row block.
  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    // input count to ReduceScatter is rows_per_rank*N * world (total reduced) / world done by API
    CHECK_NCCL(ncclReduceScatter(Cpartial[r],
                                 Cstrip[r],
                                 rows_per_rank*N,
                                 ncclFloat,
                                 ncclSum,
                                 comms[r],
                                 streams[r]));
  }

  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(streams[r]));
  }

  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaFree(A[r]));
    CHECK_CUDA(cudaFree(B[r]));
    CHECK_CUDA(cudaFree(Cpartial[r]));
    CHECK_CUDA(cudaFree(Cstrip[r]));
    CHECK_CUDA(cudaStreamDestroy(streams[r]));
    ncclCommDestroy(comms[r]);
  }
  printf("GEMM + ReduceScatter: OK\n");
  return 0;
}
