// I NEVER TESTED THIS FILE.

#include "solve.h"
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define CEIL_DIV(dividend, divisor) ((dividend + divisor - 1) / divisor)
#define COARSENING_FACTOR 2

__global__ void matrix_multiplication_kernel(float const * const A,
                                             float const * const B,
                                             float * const C,
                                             int const M,
                                             int const K,
                                             int const N) {
    // In this function K is the contraction dimension.

    __shared__ float AShared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float BShared[TILE_WIDTH][TILE_WIDTH];

    int const outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int const outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0.0;
    for (int tileIndex = 0; tileIndex < CEIL_DIV(K, TILE_WIDTH); tileIndex++) {
        // Load shared memory tile.
        if (outputRow < M && tileIndex * TILE_WIDTH + threadIdx.x < K) {
            AShared[threadIdx.y][threadIdx.x] = A[outputRow * K + tileIndex * TILE_WIDTH + threadIdx.x];
        } else {
            AShared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Apply corner turning to B
        if (tileIndex * TILE_WIDTH + threadIdx.y < K && outputCol < N) {
            BShared[threadIdx.x][threadIdx.y] = B[outputCol * K + tileIndex * TILE_WIDTH + threadIdx.y];
        } else {
            BShared[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int offsetInTile = 0; offsetInTile < TILE_WIDTH; offsetInTile++) {
            result += AShared[threadIdx.y][offsetInTile] * BShared[offsetInTile][threadIdx.x];
        }

        __syncthreads();
    }
    if (outputRow < M && outputCol < N) {
        C[outputRow * N + outputCol] = result;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(float const* const A, float const* const B, float* const C, int M, int K, int N) {
    // In this function K is the contraction dimension.
    dim3 const threadsPerBlock = dim3(16, 16);
    dim3 const blocksPerGrid = dim3(CEIL_DIV(N, threadsPerBlock.x),
                                    CEIL_DIV(M, threadsPerBlock.y));
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
}
