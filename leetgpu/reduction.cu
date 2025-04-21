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


unsigned int const maxInputsPerSolveCall = 2048;

__global__ void sum_kernel(float const* const input,
                           float* const output,
                           int const N,
                           float* const buffer) {
    // X is the offset into buffer, not into input.
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const bufferLength = (N + 1) / 2;

    if (x < bufferLength) {
        buffer[x] = input[2 * x] + (2 * x + 1 < N ? input[2 * x + 1] : 0);
    }
    __syncthreads();

    for (int distanceFromNeighborToSum = 1;
         distanceFromNeighborToSum < bufferLength;
         distanceFromNeighborToSum *= 2) {
        if (x % (2 * distanceFromNeighborToSum) == 0) {
            buffer[x] = buffer[x] + (x + distanceFromNeighborToSum < bufferLength ? buffer[x + distanceFromNeighborToSum] : 0);
        }
        __syncthreads();
    }
    if (x == 0) {
        *output = buffer[0];
    }
}

// input, output are device pointers
void solve(float const* const input,
           float* const output,
           int const N) {  
    if (N <= maxInputsPerSolveCall) {
        float* buffer;
        gpuErrchk(cudaMalloc((void**)&buffer, ((N + 1) / 2) * sizeof(float)));
        dim3 const threadsPerBlock = dim3(1024);
        // There should always be only 1 block.
        dim3 const blocksPerGrid = dim3(CEIL_DIV(N, (2 * threadsPerBlock.x)));
        sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, buffer);
    } else {
        unsigned int const numSolveCalls = CEIL_DIV(N, maxInputsPerSolveCall); 
        float* solveCallOutputs;
        gpuErrchk(cudaMalloc((void**)&solveCallOutputs, numSolveCalls * sizeof(float)));

        dim3 const threadsPerBlock = dim3(1024);
        dim3 const blocksPerGrid = dim3(1);
        for (int numProcessedItems = 0; numProcessedItems < N; numProcessedItems += maxInputsPerSolveCall) {
            unsigned int callNumber = numProcessedItems / maxInputsPerSolveCall;
            unsigned int const numItemsForCall = min(maxInputsPerSolveCall, N - numProcessedItems);
            float* buffer;
            gpuErrchk(cudaMalloc((void**)&buffer, ((numItemsForCall + 1) / 2) * sizeof(float)));
            sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input + numProcessedItems,
                                                           solveCallOutputs + callNumber,
                                                           numItemsForCall,
                                                           buffer);
            gpuErrchk(cudaFree(buffer));
        }
        solve(solveCallOutputs, output, numSolveCalls);
        gpuErrchk(cudaFree(solveCallOutputs));
    }
}
