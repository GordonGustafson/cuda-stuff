#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32
#define MAX_KERNEL_AREA (16 * 1024)
#define cdiv(dividend, divisor) ((dividend + divisor - 1) / dividend)

__constant__ float kernel_constant[MAX_KERNEL_AREA];

__global__ void convolution_2d(float const* const input,
                    float* const output,
                    int const input_rows,
                    int const input_cols,
                    int const kernel_rows,
                    int const kernel_cols) {
    int const row = blockIdx.y * blockDim.y + threadIdx.y;
    int const col = blockIdx.x * blockDim.x + threadIdx.x;

    int const output_rows = input_rows - kernel_rows + 1;
    int const output_cols = input_cols - kernel_cols + 1;

    __shared__ float input_tile_shared[BLOCK_SIZE][BLOCK_SIZE];

    if (row < input_rows && col < input_cols) {
        input_tile_shared[threadIdx.y][threadIdx.x] = input[row * input_cols + col];
    }
    __syncthreads();

    if (row < output_rows && col < output_cols) {
        float result = 0.0f;
        for (int kernel_row = 0; kernel_row < kernel_rows; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_cols; kernel_col++) {
                result += input_tile_shared[row + kernel_row][col + kernel_col] * kernel_constant[kernel_row * kernel_cols + kernel_col];
            }
        }

        output[row * output_cols + col] = result;
    }
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int const kernel_area = kernel_rows * kernel_cols;
    if (kernel_area > MAX_KERNEL_AREA) {
        printf("Kernel is larger than MAX_KERNEL_AREA constant");
        return;
    }
    cudaMemcpyToSymbol(kernel_constant, kernel, kernel_area * sizeof(float));

    dim3 const threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 const blocksPerGrid = dim3(cdiv(input_rows, threadsPerBlock.y),
                                    cdiv(input_cols, threadsPerBlock.x));
    convolution_2d<<<blocksPerGrid, threadsPerBlock>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
