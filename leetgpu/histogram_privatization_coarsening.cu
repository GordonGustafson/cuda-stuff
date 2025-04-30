#include "solve.h"
#include <cuda_runtime.h>

#define COARSENING_FACTOR 4

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

__global__ void histogram_kernel(int const* const input,
                                 int* const histogram,
                                 int const N,
                                 int const num_bins) {
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int histogram_shared[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        histogram_shared[i] = 0;
    }

    __syncthreads();
    
    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        atomicAdd(&(histogram_shared[input[i]]), 1);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        int const localBinValue = histogram_shared[i];
        if (localBinValue > 0) {
            atomicAdd(&(histogram[i]), localBinValue);
        }
    }
}

// input, histogram are device pointers
void solve(int const* const input,
           int* const histogram,
           int const N,
           int const num_bins) {
    dim3 const threadsPerBlock = dim3(1024);
    dim3 const blocksPerGrid = dim3(CEIL_DIV(N, threadsPerBlock.x * COARSENING_FACTOR));
    int const sharedMemory = num_bins * sizeof(int);
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemory>>>(input, histogram, N, num_bins);
}
