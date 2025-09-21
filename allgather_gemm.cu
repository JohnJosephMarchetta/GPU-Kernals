
#pragma once
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
int main() {
  const int world = 8;
  int devs[world]; for (int i=0;i<world;++i) devs[i]=i;
  ncclComm_t comms[world];
  CHECK_NCCL(ncclCommInitAll(comms, world, devs));

  // Sizes
  const int M = 1024, K = 512, N = 1024;
  const int n_per_rank = N / world;             // shard columns of B

  cudaStream_t streams[world];
  float *Arows[world], *Bshard[world], *Ball[world], *C[world];

  // Each rank keeps some subset of A rows too (optional). Here everyone has full A for simplicity.
  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamCreate(&streams[r]));
    CHECK_CUDA(cudaMalloc(&Arows[r], M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Bshard[r], K*n_per_rank*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Ball[r], K*N*sizeof(float)));  // after allgather
    CHECK_CUDA(cudaMalloc(&C[r], M*N*sizeof(float)));
    fill_with_index(Arows[r], M*K, r);
    fill_with_index(Bshard[r], K*n_per_rank, r);
  }

  // AllGather B shards -> Ball (concatenate along columns)
  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_NCCL(ncclAllGather(Bshard[r], Ball[r], K*n_per_rank, ncclFloat,
                             comms[r], streams[r]));
  }

  // GEMM with full B per rank (naive)
  dim3 block(16,16);
  dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    sgemm_naive<<<grid, block, 0, streams[r]>>>(M,N,K,Arows[r],Ball[r],C[r]);
  }

  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(streams[r]));
    CHECK_CUDA(cudaFree(Arows[r]));
    CHECK_CUDA(cudaFree(Bshard[r]));
    CHECK_CUDA(cudaFree(Ball[r]));
    CHECK_CUDA(cudaFree(C[r]));
    CHECK_CUDA(cudaStreamDestroy(streams[r]));
    ncclCommDestroy(comms[r]);
  }
  printf("AllGather + GEMM: OK\n");
  return 0;
}
