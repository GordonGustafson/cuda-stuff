#include "solve.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>

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

#define ALL_THREADS_IN_WARP_MASK 0xffffffffu
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32

#define COARSENING_FACTOR 13

// From https://stackoverflow.com/a/17401122
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float const* const input,
                           float* const maxBuffer,
                           int const N) {
    float __shared__ sharedBuffer[WARPS_PER_BLOCK];
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    float localMax = (index < N) ? input[index] : -INFINITY;

    int const threadsPerGrid = gridDim.x * blockDim.x;
    for (int nextElementIndex = index + threadsPerGrid; nextElementIndex < N; nextElementIndex += threadsPerGrid) {
        localMax = ::fmax(localMax, input[nextElementIndex]);
    }

    for (int numActiveThreadsInWarp = THREADS_PER_WARP / 2; numActiveThreadsInWarp >= 1; numActiveThreadsInWarp /= 2) {
        localMax = ::fmax(localMax, __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreadsInWarp));
    }
    int const warpIdx = threadIdx.x / THREADS_PER_WARP;
    int const lane = threadIdx.x % THREADS_PER_WARP;

    if (lane == 0) {
        sharedBuffer[warpIdx] = localMax;
    }
    __syncthreads();

    if (threadIdx.x < THREADS_PER_WARP) {
        localMax = sharedBuffer[lane];

        for (int numActiveThreads = THREADS_PER_WARP / 2; numActiveThreads >= 1; numActiveThreads /= 2) {
            localMax = ::fmax(localMax, __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreads));
        }
        if (threadIdx.x == 0) {
            atomicMax(maxBuffer, localMax);
        }
    }
}

__global__ void sum_exp_minus_max_kernel(float const* const input,
                                         float* const maxBuffer,
                                         float* const sumBuffer,
                                         int const N) {
    float __shared__ sharedBuffer[WARPS_PER_BLOCK];
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    float const globalMax = *maxBuffer;
    float localSum = (index < N) ? expf(input[index] - globalMax) : 0.0f;


    int const threadsPerGrid = gridDim.x * blockDim.x;
    for (int nextElementIndex = index + threadsPerGrid; nextElementIndex < N; nextElementIndex += threadsPerGrid) {
        localSum += expf(input[nextElementIndex] - globalMax);
    }

    for (int numActiveThreads = THREADS_PER_WARP / 2; numActiveThreads >= 1; numActiveThreads /= 2) {
        localSum += __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreads);
    }

    if (threadIdx.x % THREADS_PER_WARP == 0) {
        sharedBuffer[threadIdx.x / THREADS_PER_WARP] = localSum;
    }

    __syncthreads();

    if (threadIdx.x < WARPS_PER_BLOCK) {
        localSum = sharedBuffer[threadIdx.x];

        for (int numActiveThreads = THREADS_PER_WARP / 2; numActiveThreads >= 1; numActiveThreads /= 2) {
            localSum += __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreads);
        }

        if (threadIdx.x == 0) {
            atomicAdd(sumBuffer, localSum);
        }
    }
}

__global__ void softmax_kernel(float const* const input,
                               float* const output,
                               float* const maxBuffer,
                               float* const sumBuffer,
                               int const N) {
    float const globalMax = *maxBuffer;
    float const globalSum = *sumBuffer;

    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    int const threadsPerGrid = gridDim.x * blockDim.x;
    for (int nextElementIndex = index; nextElementIndex < N; nextElementIndex += threadsPerGrid) {
        float const x_i = input[nextElementIndex];
        float const exp_x_i_minus_max = expf(x_i - globalMax);
        output[nextElementIndex] = exp_x_i_minus_max / globalSum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {

    float zeroArray[1] = {0.0f};
    float negInfArray[1] = {-INFINITY};

    float* maxBuffer_d;
    gpuErrchk(cudaMalloc((void**)&maxBuffer_d, 1 * sizeof(float)));
    gpuErrchk(cudaMemcpy(maxBuffer_d, negInfArray, 1 * sizeof(float), cudaMemcpyHostToDevice));

    float* sumBuffer_d;
    gpuErrchk(cudaMalloc((void**)&sumBuffer_d, 1 * sizeof(float)));
    gpuErrchk(cudaMemcpy(sumBuffer_d, zeroArray, 1 * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = CEIL_DIV(N, threadsPerBlock * COARSENING_FACTOR);
    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, maxBuffer_d, N);
    sum_exp_minus_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, maxBuffer_d, sumBuffer_d, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, maxBuffer_d, sumBuffer_d, N);
    cudaDeviceSynchronize();
}

#define ARRAY_SIZE 500000

int main() {
    float* inputArray = new float[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        inputArray[i] = (float)i;
    }

    float* inputArray_d;
    gpuErrchk(cudaMalloc((void**)&inputArray_d, ARRAY_SIZE * sizeof(float)));
    gpuErrchk(cudaMemcpy(inputArray_d, inputArray, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    float* outputArray_d;
    gpuErrchk(cudaMalloc((void**)&outputArray_d, ARRAY_SIZE * sizeof(float)));

    solve(inputArray_d, outputArray_d, ARRAY_SIZE);

    gpuErrchk(cudaFree(inputArray_d));
    gpuErrchk(cudaFree(outputArray_d));
    delete[] inputArray;

    return 0;
}
