#include "solve.h"
#include <cuda_runtime.h>

int constexpr INPUT_TILE_NUM_ROWS = 64;
int constexpr INPUT_TILE_NUM_COLS = 64;

__host__ __device__ int constexpr ceil_div(int const dividend, int const divisor) {
    return (dividend + divisor - 1) / divisor;
}

__global__ void gaussian_blur_kernel(const float* const input_hbm,
                                     const float* const kernel_hbm,
                                     float* const output_hbm,
                                     int const input_num_rows,
                                     int const input_num_cols,
                                     int const kernel_num_rows,
                                     int const kernel_num_cols,
                                     int const output_tile_num_rows,
                                     int const output_tile_num_cols) {
    __shared__ float input_shm[INPUT_TILE_NUM_ROWS][INPUT_TILE_NUM_COLS];

    int const output_shm_top_row_hbm = blockIdx.y * output_tile_num_rows;
    int const output_shm_left_col_hbm = blockIdx.x * output_tile_num_cols;

    // load input_shm
    for (int input_col_shm = threadIdx.x; input_col_shm < INPUT_TILE_NUM_COLS; input_col_shm += blockDim.x) {
        for (int input_row_shm = threadIdx.y; input_row_shm < INPUT_TILE_NUM_ROWS; input_row_shm += blockDim.y) {
            int const input_col_hbm = output_shm_left_col_hbm + input_col_shm - kernel_num_cols / 2;
            int const input_row_hbm = output_shm_top_row_hbm + input_row_shm - kernel_num_rows / 2;
            bool const input_col_hbm_in_bounds = input_col_hbm >= 0 && input_col_hbm < input_num_cols;
            bool const input_row_hbm_in_bounds = input_row_hbm >= 0 && input_row_hbm < input_num_rows;
            if (input_col_hbm_in_bounds && input_row_hbm_in_bounds) {
                input_shm[input_row_shm][input_col_shm] = input_hbm[input_row_hbm * input_num_cols + input_col_hbm];
            } else {
                input_shm[input_row_shm][input_col_shm] = 0.0f;
            }
        }
    }

    __syncthreads();  // Wait until we've loaded input_shm before reading it

    // compute and write result
    for (int output_col_shm = threadIdx.x; output_col_shm < output_tile_num_cols; output_col_shm += blockDim.x) {
        for (int output_row_shm = threadIdx.y; output_row_shm < output_tile_num_rows; output_row_shm += blockDim.y) {
            float result = 0.0f;
            int const input_col_shm = output_col_shm + kernel_num_cols / 2;
            int const input_row_shm = output_row_shm + kernel_num_rows / 2;

            for (int col_offset = - kernel_num_cols / 2; col_offset <= kernel_num_cols / 2; col_offset++) {
                for (int row_offset = - kernel_num_rows / 2; row_offset <= kernel_num_rows / 2; row_offset++) {
                    int const input_col_with_offset_shm = input_col_shm + col_offset;
                    int const input_row_with_offset_shm = input_row_shm + row_offset;
                    result += input_shm[input_row_with_offset_shm][input_col_with_offset_shm] * kernel_hbm[(row_offset + kernel_num_rows / 2) * kernel_num_cols + col_offset + kernel_num_cols / 2];
                }
            }

            int const output_row_hbm = output_shm_top_row_hbm + output_row_shm;
            int const output_col_hbm = output_shm_left_col_hbm + output_col_shm;
            bool const output_row_hbm_in_bounds = output_row_hbm >= 0 && output_row_hbm < input_num_rows;
            bool const output_col_hbm_in_bounds = output_col_hbm >= 0 && output_col_hbm < input_num_cols;
            if (output_row_hbm_in_bounds && output_col_hbm_in_bounds) {
                output_hbm[output_row_hbm * input_num_cols + output_col_hbm] = result;
            }
        }
    }
}

// input, kernel, output are device pointers
void solve(const float* const input,
           const float* const kernel,
           float* const output,
           int const input_rows,
           int const input_cols,
           int const kernel_rows,
           int const kernel_cols) {

    int const output_tile_num_rows = INPUT_TILE_NUM_ROWS - kernel_rows + 1;
    int const output_tile_num_cols = INPUT_TILE_NUM_COLS - kernel_cols + 1;

    dim3 const threadsPerBlock = dim3(32, 32);
    dim3 const blocksPerGrid = dim3(ceil_div(input_cols, output_tile_num_cols),
                                    ceil_div(input_rows, output_tile_num_rows));
    gaussian_blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, output_tile_num_rows, output_tile_num_cols);
}
