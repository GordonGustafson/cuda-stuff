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

unsigned int const maxThreadsPerBlock = 1024;

__global__ void solve_kernel(float const* const input,
                             float* const output,
                             int const N) {  
    extern __shared__ float sharedBuffer[];
    sharedBuffer[threadIdx.x] = 0.0f;

    int const threadsPerGrid = gridDim.x * blockDim.x;
    int const firstElementIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (firstElementIdx < N) {
        int const secondElementIndex = firstElementIdx + threadsPerGrid;
        sharedBuffer[threadIdx.x] = input[firstElementIdx] + (secondElementIndex < N ? input[secondElementIndex] : 0);
    }

    for (int numActiveThreads = blockDim.x / 2; numActiveThreads >= 1; numActiveThreads /= 2) {
        __syncthreads();
        if (threadIdx.x < numActiveThreads) {
            sharedBuffer[threadIdx.x] += sharedBuffer[threadIdx.x + numActiveThreads];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, sharedBuffer[0]);
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = CEIL_DIV(N, threadsPerBlock * 2);
    solve_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
}
