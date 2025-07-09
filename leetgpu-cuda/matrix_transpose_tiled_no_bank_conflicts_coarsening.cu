#include "solve.h"
#include <cuda_runtime.h>

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

#define BLOCK_SIZE 32
// There's 48KB of non-dynamic shared memory allowed per block.
// We can't fit a value of 12 due to the BLOCK_SIZE+1 that avoids shared memory bank conflicts.
#define COARSENING_FACTOR 11

__global__ void matrix_transpose_kernel(const float* input, float* output, int inputRows, int inputCols) {
    __shared__ float blockShared[COARSENING_FACTOR*BLOCK_SIZE][BLOCK_SIZE+1];
    int const inputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int const inputRow = blockIdx.y * blockDim.y * COARSENING_FACTOR + threadIdx.y;
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        bool const inputInBounds = inputCol < inputCols && (i * blockDim.y + inputRow) < inputRows;
        if (inputInBounds) {
            blockShared[i * blockDim.y + threadIdx.y][threadIdx.x] = input[(i * blockDim.y + inputRow) * inputCols + inputCol];
        }
    }

    __syncthreads();

    int const outputCol = blockIdx.y * blockDim.y * COARSENING_FACTOR + threadIdx.x;
    int const outputRow = blockIdx.x * blockDim.x + threadIdx.y;
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        bool const outputInBounds = outputCol + i * blockDim.y < inputRows && outputRow < inputCols;
        if (outputInBounds) {
            output[outputRow * inputRows + outputCol + i * blockDim.y] = blockShared[threadIdx.x + i * blockDim.y][threadIdx.y];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(CEIL_DIV(cols, BLOCK_SIZE),
                       CEIL_DIV(rows, BLOCK_SIZE * COARSENING_FACTOR));
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
}
