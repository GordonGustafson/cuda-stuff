#include "solve.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

#define ELEMENTS_PER_THREAD 4
unsigned int const maxThreadsPerBlock = 1024;
#define ALL_THREADS_IN_WARP_MASK 0xffffffffU
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32

__global__ void solve_kernel(float const* const input,
                             float* const output,
                             int const N) {  
    extern __shared__ float sharedBuffer[WARPS_PER_BLOCK];

    float localSum = 0.0f;
    int const threadsPerGrid = gridDim.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (index < N) {
            localSum += input[index];
            index += threadsPerGrid;
        }
    }

    for (int numActiveThreadsInWarp = THREADS_PER_WARP / 2; numActiveThreadsInWarp >= 1; numActiveThreadsInWarp /= 2) {
        localSum += __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreadsInWarp);
    }

    int const warpIndex = threadIdx.x / THREADS_PER_WARP;
    int const lane = threadIdx.x % THREADS_PER_WARP;
    if (lane == 0) {
        sharedBuffer[warpIndex] = localSum;
    }
    __syncthreads();
    if (threadIdx.x < WARPS_PER_BLOCK) {
        localSum = sharedBuffer[threadIdx.x];
        for (int numActiveThreads = WARPS_PER_BLOCK / 2; numActiveThreads >= 1; numActiveThreads /= 2) {
            localSum += __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreads);
        }

        if (threadIdx.x == 0) {
            atomicAdd(output, localSum);
        }
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = CEIL_DIV(N, threadsPerBlock * ELEMENTS_PER_THREAD);
    solve_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
