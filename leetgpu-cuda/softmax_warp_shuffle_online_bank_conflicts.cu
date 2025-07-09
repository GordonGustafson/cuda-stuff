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

#define MAX_INDEX 0
#define SUM_INDEX 1

typedef union {
    float pair[2];
    unsigned long long int ulong;    // for atomic update
} atomicFloatPair;

__device__ static inline float onlineSoftmaxSum(float const maxA,
                                                float const sumA,
                                                float const maxB,
                                                float const sumB) {
    if (sumA == 0.0f) {
        return sumB;
    } else if (sumB == 0.0f) {
        return sumA;
    } else if (maxA > maxB) {
        return sumB * expf(maxB - maxA) + sumA;
    } else {
        return sumB + sumA * expf(maxA - maxB);
    }
}

__device__ static unsigned long long int atomicSoftmaxUpdateMaxAndSum(unsigned long long int* address, float localMax, float localSum) {
    atomicFloatPair currentGlobalPair;
    unsigned long long currentGlobalUlong = *address;
    do {
        currentGlobalPair.ulong = currentGlobalUlong;
        float const currentGlobalMax = currentGlobalPair.pair[MAX_INDEX];
        float const currentGlobalSum = currentGlobalPair.pair[SUM_INDEX];
        atomicFloatPair newGlobalPair;
        newGlobalPair.pair[MAX_INDEX] = ::fmax(localMax, currentGlobalMax);
        newGlobalPair.pair[SUM_INDEX] = onlineSoftmaxSum(localMax, localSum, currentGlobalMax, currentGlobalSum);
        currentGlobalUlong = atomicCAS(address, currentGlobalPair.ulong,  newGlobalPair.ulong);
    } while (currentGlobalUlong != currentGlobalPair.ulong);
    return currentGlobalUlong;
}

__global__ void max_and_sum_kernel(float const* const input,
                                   unsigned long long int* const maxAndSumBuffer,
                                   int const N) {
    float __shared__ sharedMax[WARPS_PER_BLOCK];
    float __shared__ sharedSumWithBankConflicts[WARPS_PER_BLOCK+1];
    float* sharedSum = &sharedSumWithBankConflicts[1];
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    float localMax = (i < N) ? input[i] : -INFINITY;
    // float localSum = (i < N) ? expf(input[i] - localMax) : 0.0f;
    float localSum = (i < N) ? 1.0f : 0.0f;

    for (int numActiveThreadsInWarp = THREADS_PER_WARP / 2; numActiveThreadsInWarp >= 1; numActiveThreadsInWarp /= 2) {
        float const incomingMax = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreadsInWarp);
        float const incomingSum = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreadsInWarp);
        localSum = onlineSoftmaxSum(localMax, localSum, incomingMax, incomingSum);
        localMax = ::fmax(localMax, incomingMax);
    }
    int const warpIdx = threadIdx.x / THREADS_PER_WARP;
    int const lane = threadIdx.x % THREADS_PER_WARP;

    if (lane == 0) {
        sharedMax[warpIdx] = localMax;
        sharedSum[warpIdx] = localSum;
    }
    __syncthreads();

    if (threadIdx.x < THREADS_PER_WARP) {
        localMax = sharedMax[lane];
        localSum = sharedSum[lane];

        for (int numActiveThreadsInWarp = THREADS_PER_WARP / 2; numActiveThreadsInWarp >= 1; numActiveThreadsInWarp /= 2) {
            float const incomingMax = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreadsInWarp);
            float const incomingSum = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreadsInWarp);
            localSum = onlineSoftmaxSum(localMax, localSum, incomingMax, incomingSum);
            localMax = ::fmax(localMax, incomingMax);
        }

        if (threadIdx.x == 0) {
            atomicSoftmaxUpdateMaxAndSum(maxAndSumBuffer, localMax, localSum);
        }
    }
}

__global__ void softmax_kernel(float const* const input,
                               float* const output,
                               unsigned long long int* const maxAndSumBuffer,
                               int const N) {
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicFloatPair maxAndSum;
        maxAndSum.ulong = *maxAndSumBuffer;
        float const globalMax = maxAndSum.pair[MAX_INDEX];
        float const globalSum = maxAndSum.pair[SUM_INDEX];
        float const x_i = input[i];
        float const exp_x_i_minus_max = expf(x_i - globalMax);
        output[i] = exp_x_i_minus_max / globalSum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {

    atomicFloatPair maxAndSum;
    maxAndSum.pair[MAX_INDEX] = -INFINITY;
    maxAndSum.pair[SUM_INDEX] = 0.0f;

    unsigned long long int* maxAndSumBuffer_d;
    gpuErrchk(cudaMalloc((void**)&maxAndSumBuffer_d, sizeof(unsigned long long int)));
    gpuErrchk(cudaMemcpy(maxAndSumBuffer_d, &maxAndSum, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = CEIL_DIV(N, threadsPerBlock);
    max_and_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, maxAndSumBuffer_d, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, maxAndSumBuffer_d, N);
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
