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

    __shared__ float AShared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float BShared[COARSENING_FACTOR][TILE_WIDTH][TILE_WIDTH];
    
    int const firstOutputCol = blockIdx.x * blockDim.x * COARSENING_FACTOR + threadIdx.x;
    int const outputRow = blockIdx.y * blockDim.y + threadIdx.y;

    float result[COARSENING_FACTOR];
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        result[i] = 0.0f;
    }

    for (int tileIndex = 0; tileIndex < CEIL_DIV(K, TILE_WIDTH); tileIndex++) {
        // Load tiles
        if (outputRow < M && tileIndex * TILE_WIDTH + threadIdx.x < K) {
            AShared[threadIdx.y][threadIdx.x] = A[outputRow * K + tileIndex * TILE_WIDTH + threadIdx.x];
        } else {
            AShared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        for (int i = 0; i < COARSENING_FACTOR; i++) {
            if (tileIndex * TILE_WIDTH + threadIdx.y < K && firstOutputCol + i * TILE_WIDTH < N) {
                BShared[i][threadIdx.y][threadIdx.x] = B[(tileIndex * TILE_WIDTH + threadIdx.y) * N + firstOutputCol + i * TILE_WIDTH];
            } else {
                BShared[i][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();

        // Multiply tiles
        for (int indexInTile = 0; indexInTile < TILE_WIDTH; indexInTile++) {
            for (int i = 0; i < COARSENING_FACTOR; i++) {
                result[i] += AShared[threadIdx.y][indexInTile] * BShared[i][indexInTile][threadIdx.x];
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < COARSENING_FACTOR; i++) {
        if (outputRow < M && firstOutputCol + i * TILE_WIDTH < N) {
            C[outputRow * N + firstOutputCol + i * TILE_WIDTH] = result[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(float const* const A, float const* const B, float* const C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / (threadsPerBlock.x * COARSENING_FACTOR),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
