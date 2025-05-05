#include "solve.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 8
#define MAX_KERNEL_VOLUME (16 * 1024)
#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

__constant__ float kernel_constant[MAX_KERNEL_VOLUME];

__global__ void convolution_3d(float const* const input,
                               float* const output,
                               int const input_depth,
                               int const input_rows,
                               int const input_cols,
                               int const kernel_depth,
                               int const kernel_rows,
                               int const kernel_cols) {

    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;
    int const z = blockIdx.z * blockDim.z + threadIdx.z;

    int const output_depth = input_depth - kernel_depth + 1;
    int const output_rows = input_rows - kernel_rows + 1;
    int const output_cols = input_cols - kernel_cols + 1;
    bool const is_output_thread = z < output_depth && y < output_rows && x < output_cols;

    __shared__ float inputShared[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

    if (x < input_cols
        && y < input_rows
        && z < input_depth) {
        inputShared[z][y][x] = input[z * input_rows * input_cols
                                     + y * input_cols
                                     + x];
    }

    __syncthreads();

    if (is_output_thread) {
        float result = 0.0f;
        for (int k_z = 0; k_z < kernel_depth; k_z++) {
            for (int k_y = 0; k_y < kernel_rows; k_y++) {
                for (int k_x = 0; k_x < kernel_cols; k_x++) {
                    float const kernel_value = kernel_constant[k_z * kernel_rows * kernel_cols
                                                               + k_y * kernel_cols
                                                               + k_x];
                    if (threadIdx.z + k_z < TILE_WIDTH
                        && threadIdx.y + k_y < TILE_WIDTH
                        && threadIdx.x + k_x < TILE_WIDTH) {
                        result += kernel_value * inputShared[threadIdx.z + k_z][threadIdx.y + k_y][threadIdx.x + k_x];
                    } else {
                        result += kernel_value * input[(z + k_z) * input_rows * input_cols
                                                       + (y + k_y) * input_cols
                                                       + (x + k_x)];
                    }
                }
            }
        }

        output[z * output_rows * output_cols
               + y * output_cols
               + x] = result;
    }
}

// input, kernel, output are device pointers
void solve(float const* const input,
           float const* const kernel,
           float* const output,
           int const input_depth,
           int const input_rows,
           int const input_cols,
           int const kernel_depth,
           int const kernel_rows,
           int const kernel_cols) {
    int const kernel_volume = kernel_depth * kernel_rows * kernel_cols;
    if (kernel_volume > MAX_KERNEL_VOLUME) {
        printf("kernel too large to fit in constant memory!");
        return;
    }

    cudaError_t const err = cudaMemcpyToSymbol(kernel_constant, kernel, kernel_volume * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return;
    }

    int const output_depth = input_depth - kernel_depth + 1;
    int const output_rows = input_rows - kernel_rows + 1;
    int const output_cols = input_cols - kernel_cols + 1;

    dim3 const threadsPerBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 const blocksPerGrid(CEIL_DIV(output_cols, threadsPerBlock.x),
                             CEIL_DIV(output_rows, threadsPerBlock.y),
                             CEIL_DIV(output_depth, threadsPerBlock.z));
    convolution_3d<<<blocksPerGrid, threadsPerBlock>>>(input, output, input_depth, input_rows, input_cols,
                                                       kernel_depth, kernel_rows, kernel_cols);
}
