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

#define TILE_WIDTH 16
#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))
#define COARSENING_FACTOR 2

enum Transpose {
    Transposed,
    Untransposed
};

#define BLOCK_SIZE 32
// There's 64 of dynamic shared memory allowed per SM in a T4 GPU.
// We can't fit a value of 16 due to the BLOCK_SIZE+1 that avoids shared memory bank conflicts.
#define TRANPOSE_COARSENING_FACTOR 15

__global__ void matrix_transpose_kernel(const float* input, float* output, int inputRows, int inputCols) {
    // __shared__ float blockShared[TRANPOSE_COARSENING_FACTOR*BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ extern float blockShared[];
    int const colsInBlockShared = BLOCK_SIZE + 1;
    int const inputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int const inputRow = blockIdx.y * blockDim.y * TRANPOSE_COARSENING_FACTOR + threadIdx.y;
#pragma unroll
    for (int i = 0; i < TRANPOSE_COARSENING_FACTOR; i++) {
        bool const inputInBounds = inputCol < inputCols && (i * blockDim.y + inputRow) < inputRows;
        if (inputInBounds) {
            blockShared[(i * blockDim.y + threadIdx.y) * colsInBlockShared + threadIdx.x] = input[(i * blockDim.y + inputRow) * inputCols + inputCol];
        }
    }

    __syncthreads();

    int const outputCol = blockIdx.y * blockDim.y * TRANPOSE_COARSENING_FACTOR + threadIdx.x;
    int const outputRow = blockIdx.x * blockDim.x + threadIdx.y;
#pragma unroll
    for (int i = 0; i < TRANPOSE_COARSENING_FACTOR; i++) {
        bool const outputInBounds = outputCol + i * blockDim.y < inputRows && outputRow < inputCols;
        if (outputInBounds) {
            output[outputRow * inputRows + outputCol + i * blockDim.y] = blockShared[(threadIdx.x + i * blockDim.y) * colsInBlockShared + threadIdx.y];
        }
    }
}

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

    for (int i = 0; i < COARSENING_FACTOR; i++) {
        if (outputRow < M && firstOutputCol + i * TILE_WIDTH < N) {
            C[outputRow * N + firstOutputCol + i * TILE_WIDTH] = result[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(float const* const A, float const* const B, float* const C, int M, int K, int N) {
    float* BTranspose;
    gpuErrchk(cudaMalloc(&BTranspose, K * N * sizeof(float)));
    dim3 threadsPerBlockTranspose(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGridTranspose(CEIL_DIV(N, BLOCK_SIZE),
                                CEIL_DIV(K, BLOCK_SIZE * TRANPOSE_COARSENING_FACTOR));
    int sharedMemory = TRANPOSE_COARSENING_FACTOR * BLOCK_SIZE * (BLOCK_SIZE+1) * sizeof(float);
    cudaFuncSetAttribute(matrix_transpose_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemory);
    matrix_transpose_kernel<<<blocksPerGridTranspose, threadsPerBlockTranspose, sharedMemory>>>(B, BTranspose, K, N);
    
    // K is the contraction dimension.
    dim3 threadsPerBlockMul(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGridMul(CEIL_DIV(N, threadsPerBlockMul.x * COARSENING_FACTOR),
                          CEIL_DIV(M, threadsPerBlockMul.y));
    matrix_multiplication_kernel<Transpose::Transposed><<<blocksPerGridMul, threadsPerBlockMul>>>(A, BTranspose, C, M, K, N);
}
