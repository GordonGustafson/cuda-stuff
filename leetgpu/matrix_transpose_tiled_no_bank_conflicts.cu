#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int inputRows, int inputCols) {
    __shared__ float blockShared[BLOCK_SIZE][BLOCK_SIZE+1];
    int const inputCol = blockIdx.x * blockDim.x + threadIdx.x;
    int const inputRow = blockIdx.y * blockDim.y + threadIdx.y;

    bool const inputInBounds = inputCol < inputCols && inputRow < inputRows;
    if (inputInBounds) {
        blockShared[threadIdx.y][threadIdx.x] = input[inputRow * inputCols + inputCol];
    }

    __syncthreads();

    int const outputCol = blockIdx.y * blockDim.y + threadIdx.x;
    int const outputRow = blockIdx.x * blockDim.x + threadIdx.y;
    bool const outputInBounds = outputCol < inputRows && outputRow < inputCols;
    if (outputInBounds) {
        output[outputRow * inputRows + outputCol] = blockShared[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
}
