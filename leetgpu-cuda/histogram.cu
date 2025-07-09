#include "solve.h"
#include <cuda_runtime.h>

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

__global__ void histogram_kernel(int const* const input,
                                 int* const histogram,
                                 int const N,
                                 int const num_bins) {
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < N) {
        atomicAdd(&(histogram[input[x]]), 1);
    }
}

// input, histogram are device pointers
void solve(int const* const input,
           int* const histogram,
           int const N,
           int const num_bins) {
    dim3 const threadsPerBlock = dim3(1024);
    dim3 const blocksPerGrid = dim3(CEIL_DIV(N, threadsPerBlock.x));
    histogram_kernel<<<threadsPerBlock, blocksPerGrid>>>(input, histogram, N, num_bins);
}
