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
int main() {
  const int world = 8;
  int devs[world]; for (int i=0;i<world;++i) devs[i]=i;

  ncclComm_t comms[world];
  CHECK_NCCL(ncclCommInitAll(comms, world, devs)); // RCCL backend on ROCm

  hipStream_t streams[world];
  float *sendbuf[world], *recvbuf[world];

  const int elems_total = 1<<16;
  const int elems_chunk = elems_total / world;

  for (int r=0;r<world;++r) {
    CHECK_HIP(hipSetDevice(r));
    CHECK_HIP(hipStreamCreate(&streams[r]));
    CHECK_HIP(hipMalloc(&sendbuf[r], elems_total*sizeof(float)));
    CHECK_HIP(hipMalloc(&recvbuf[r], elems_total*sizeof(float)));
    fill_with_index(sendbuf[r], elems_total, r);
  }

  for (int r=0;r<world;++r) {
    CHECK_HIP(hipSetDevice(r));
    CHECK_NCCL(ncclGroupStart());
    for (int peer=0; peer<world; ++peer) {
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
    CHECK_HIP(hipSetDevice(r));
    CHECK_HIP(hipStreamSynchronize(streams[r]));
  }

  for (int r=0;r<world;++r) {
    CHECK_HIP(hipFree(sendbuf[r]));
    CHECK_HIP(hipFree(recvbuf[r]));
    CHECK_HIP(hipStreamDestroy(streams[r]));
    ncclCommDestroy(comms[r]);
  }
  printf("All-to-all (HIP/RCCL): OK\n");
  return 0;
}
