#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include "helpers.h"


#define CHECK_CUDA(call) do {                                 
  cudaError_t _e = (call);                                     
  if (_e != cudaSuccess) {                                     
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,    
            cudaGetErrorString(_e));                           
    std::exit(1);                                              
  }                                                            
} while(0)

#define CHECK_NCCL(call) do {                                  
  ncclResult_t _e = (call);                                     
  if (_e != ncclSuccess) {                                     
    fprintf(stderr, "NCCL %s:%d: %s\n", __FILE__, __LINE__,    
            ncclGetErrorString(_e));                           
    std::exit(1);                                              
  }                                                            
} while(0)

inline void fill_with_index(float* p, int n, int gpu) {
  // quick host fill (debug); normally you'd cudaMemsetAsync or a kernel
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = gpu*1000.0f + i;
  CHECK_CUDA(cudaMemcpy(p, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
}

int main() {
  const int world = 8;                         // number of GPUs
  int devs[world]; for (int i=0;i<world;++i) devs[i]=i;

  // one process controlling all GPUs
  ncclComm_t comms[world];
  CHECK_NCCL(ncclCommInitAll(comms, world, devs));

  // one stream per GPU
  cudaStream_t streams[world];
  float *sendbuf[world], *recvbuf[world];

  // toy sizes: total elements per GPU
  const int elems_total = 1<<16;               // divisible by world for simplicity
  const int elems_chunk = elems_total / world; // what we send to each peer

  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamCreate(&streams[r]));
    CHECK_CUDA(cudaMalloc(&sendbuf[r], elems_total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf[r], elems_total * sizeof(float)));
    fill_with_index(sendbuf[r], elems_total, r);
  }

  // All-to-all via grouped sends/recvs
  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_NCCL(ncclGroupStart());
    for (int peer=0; peer<world; ++peer) {
      // offsets measured in elements
      size_t s_off = peer * elems_chunk;
      size_t r_off = peer * elems_chunk;
      CHECK_NCCL(ncclSend(sendbuf[r] + s_off, elems_chunk, ncclFloat,
                          peer, comms[r], streams[r]));
      CHECK_NCCL(ncclRecv(recvbuf[r] + r_off, elems_chunk, ncclFloat,
                          peer, comms[r], streams[r]));
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(streams[r]));
  }

  // cleanup
  for (int r=0;r<world;++r) {
    CHECK_CUDA(cudaFree(sendbuf[r]));
    CHECK_CUDA(cudaFree(recvbuf[r]));
    CHECK_CUDA(cudaStreamDestroy(streams[r]));
    ncclCommDestroy(comms[r]);
  }
  printf("All-to-all: OK\n");
  return 0;
}


