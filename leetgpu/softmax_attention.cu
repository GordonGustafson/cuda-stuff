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
#define TILE_WIDTH 16
#define MM_COARSENING_FACTOR 2

enum Transpose {
    Transposed,
    Untransposed
};

template <Transpose T>
__global__ void matrix_multiplication_kernel(float const * const A,
                                             float const * const B,
                                             float * const C,
                                             int const M,
                                             int const K,
                                             int const N) {
    // K is the contraction dimension.
    __shared__ float AShared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float BShared[TILE_WIDTH][TILE_WIDTH];
    
    int const firstOutputCol = blockIdx.x * blockDim.x * MM_COARSENING_FACTOR + threadIdx.x;
    int const outputRow = blockIdx.y * blockDim.y + threadIdx.y;

    float result[MM_COARSENING_FACTOR];
    for (int i = 0; i < MM_COARSENING_FACTOR; i++) {
        result[i] = 0.0f;
    }

    for (int tileIndex = 0; tileIndex < CEIL_DIV(K, TILE_WIDTH); tileIndex++) {
        // Load tiles
        if (outputRow < M && tileIndex * TILE_WIDTH + threadIdx.x < K) {
            AShared[threadIdx.y][threadIdx.x] = A[outputRow * K + tileIndex * TILE_WIDTH + threadIdx.x];
        } else {
            AShared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        for (int i = 0; i < MM_COARSENING_FACTOR; i++) {
            if (tileIndex * TILE_WIDTH + threadIdx.y < K && firstOutputCol + i * TILE_WIDTH < N) {
                if (T == Transpose::Untransposed) {
                    BShared[threadIdx.y][threadIdx.x] = B[(tileIndex * TILE_WIDTH + threadIdx.y) * N + firstOutputCol + i * TILE_WIDTH];
                } else {
                    BShared[threadIdx.y][threadIdx.x] = B[(firstOutputCol + i * TILE_WIDTH) * K + tileIndex * TILE_WIDTH + threadIdx.y];
                }
            } else {
                BShared[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Multiply tiles
            for (int indexInTile = 0; indexInTile < TILE_WIDTH; indexInTile++) {
                result[i] += AShared[threadIdx.y][indexInTile] * BShared[indexInTile][threadIdx.x];
            }

            __syncthreads();
        }
    }

    for (int i = 0; i < MM_COARSENING_FACTOR; i++) {
        if (outputRow < M && firstOutputCol + i * TILE_WIDTH < N) {
            C[outputRow * N + firstOutputCol + i * TILE_WIDTH] = result[i];
        }
    }
}


#define ALL_THREADS_IN_WARP_MASK 0xffffffffu
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32

#define MAX_INDEX 0
#define SUM_INDEX 1

__device__ static inline float onlineSoftmaxSum(float const maxA,
                                                float const sumA,
                                                float const maxB,
                                                float const sumB,
                                                float const temperature) {
    if (sumA == 0.0f) {
        return sumB;
    } else if (sumB == 0.0f) {
        return sumA;
    } else if (maxA > maxB) {
        return sumB * expf((maxB / temperature) - maxA) + sumA;
    } else {
        return sumB + sumA * expf((maxA / temperature) - maxB);
    }
}

__global__ void softmax_across_cols_kernel(float const* const input,
                                           float* const output,
                                           int const cols,
                                           float const temperature) {
    float __shared__ globalMaxAndSum[2];
    float __shared__ sharedMax[WARPS_PER_BLOCK];
    float __shared__ sharedSumWithBankConflicts[WARPS_PER_BLOCK+1];
    float* sharedSum = &sharedSumWithBankConflicts[1];
    float localMax = (threadIdx.x < cols) ? input[blockIdx.x * cols + threadIdx.x] : -INFINITY;
    // float localSum = (i < cols) ? expf(x_i - localMax) : 0.0f;
    float localSum = (threadIdx.x < cols) ? 1.0f : 0.0f;

    int const threadsPerBlock = blockDim.x;
    for (int nextElementIndex = threadIdx.x + threadsPerBlock; nextElementIndex < cols; nextElementIndex += threadsPerBlock) {
        float const incomingMax = input[blockIdx.x * cols + nextElementIndex];
        float const incomingSum = 1.0f;
        localSum = onlineSoftmaxSum(localMax, localSum, incomingMax, incomingSum, temperature);
        localMax = ::fmax(localMax, incomingMax);
    }

    #pragma unroll
    for (int numActiveThreadsInWarp = THREADS_PER_WARP / 2; numActiveThreadsInWarp >= 1; numActiveThreadsInWarp /= 2) {
        float const incomingMax = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreadsInWarp);
        float const incomingSum = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreadsInWarp);
        localSum = onlineSoftmaxSum(localMax, localSum, incomingMax, incomingSum, temperature);
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

        #pragma unroll
        for (int numActiveThreadsInWarp = THREADS_PER_WARP / 2; numActiveThreadsInWarp >= 1; numActiveThreadsInWarp /= 2) {
            float const incomingMax = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreadsInWarp);
            float const incomingSum = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreadsInWarp);
            localSum = onlineSoftmaxSum(localMax, localSum, incomingMax, incomingSum, temperature);
            localMax = ::fmax(localMax, incomingMax);
        }

        if (threadIdx.x == 0) {
            globalMaxAndSum[MAX_INDEX] = localMax;
            globalMaxAndSum[SUM_INDEX] = localSum;
        }
    }

    float const globalMax = globalMaxAndSum[MAX_INDEX];
    float const globalSum = globalMaxAndSum[SUM_INDEX];

    for (int nextElementIndex = threadIdx.x; nextElementIndex < cols; nextElementIndex += threadsPerBlock) {
        float const x_i = input[blockIdx.x * cols + nextElementIndex];
        float const exp_x_i_minus_max = expf((x_i - globalMax) / temperature);
        output[blockIdx.x * cols + nextElementIndex] = exp_x_i_minus_max / globalSum;
    }
}


// Q, K, V, output are device pointers
void solve(float const* const Q,  // size Mxd
           float const* const K,  // size Nxd
           float const* const V,  // size Nxd
           float* const output,   // size Mxd
           int const M,
           int const N,
           int const d) {
    float* S;
    gpuErrchk(cudaMalloc((void**)&S, M * N * sizeof(float)));
    float* P;
    gpuErrchk(cudaMalloc((void**)&P, M * N * sizeof(float)));

    dim3 mmThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 mmBlocksPerGrid(CEIL_DIV(N, mmThreadsPerBlock.x * MM_COARSENING_FACTOR),
                         CEIL_DIV(M, mmThreadsPerBlock.y));
    matrix_multiplication_kernel<Transpose::Transposed><<<mmBlocksPerGrid, mmThreadsPerBlock>>>(Q, K, S, M, d, N);

    softmax_across_cols_kernel<<<1024, M>>>(S, P, N, sqrt((float)d));

    mmThreadsPerBlock = dim3(TILE_WIDTH, TILE_WIDTH);
    mmBlocksPerGrid = dim3(CEIL_DIV(d, mmThreadsPerBlock.x * MM_COARSENING_FACTOR),
                           CEIL_DIV(M, mmThreadsPerBlock.y));
    matrix_multiplication_kernel<Transpose::Untransposed><<<mmBlocksPerGrid, mmThreadsPerBlock>>>(P, V, output, M, N, d);
}
