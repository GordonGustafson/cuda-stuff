#include "solve.h"
#include <cuda_runtime.h>

int constexpr SHM_TILE_SIZE = 32;

__host__ __device__ int constexpr ceil_div(int const dividend, int const divisor) {
    return (dividend + divisor - 1) / divisor;
}

__global__ void batched_matrix_multiplication_kernel(const float* const A_hbm,
                                                     const float* const B_hbm,
                                                     float* const C_hbm,
                                                     int const BATCH,
                                                     int const M,
                                                     int const N,
                                                     int const K) {
    __shared__ float A_shm[SHM_TILE_SIZE][SHM_TILE_SIZE];
    __shared__ float B_shm[SHM_TILE_SIZE][SHM_TILE_SIZE];

    int const shm_tile_top_row_hbm = blockIdx.y * SHM_TILE_SIZE;
    int const shm_tile_left_column_hbm = blockIdx.x * SHM_TILE_SIZE;
    int const thread_output_row_hbm = shm_tile_top_row_hbm + threadIdx.y;
    int const thread_output_col_hbm = shm_tile_left_column_hbm + threadIdx.x;

    float result = 0.0f;
    for (int contraction_tile = 0; contraction_tile < ceil_div(K, SHM_TILE_SIZE); contraction_tile++) {
        // Load A and B 
        int const A_row_hbm = thread_output_row_hbm;
        int const A_col_hbm = contraction_tile * SHM_TILE_SIZE + threadIdx.x;
        int const B_row_hbm = contraction_tile * SHM_TILE_SIZE + threadIdx.y;
        int const B_col_hbm = thread_output_col_hbm;

        if (A_row_hbm < M && A_col_hbm < K) {
            A_shm[threadIdx.y][threadIdx.x] = A_hbm[blockIdx.z * M * K + A_row_hbm * K + A_col_hbm];
        }
        if (B_row_hbm < K && B_col_hbm < N) {
            B_shm[threadIdx.y][threadIdx.x] = B_hbm[blockIdx.z * K * N + B_row_hbm * N + B_col_hbm];
        }

        __syncthreads();  // Don't read shm_tile until we're done writing it.

        for (int index = 0;
             index < SHM_TILE_SIZE && shm_tile_top_row_hbm + index < K && shm_tile_left_column_hbm + index < K;
             index++) {
            result += A_shm[threadIdx.y][index] * B_shm[index][threadIdx.x];
        }

        __syncthreads();  // Don't write shm_tile until we're done reading it.
    }

    if (thread_output_row_hbm < M && thread_output_col_hbm < N) {
        C_hbm[blockIdx.z * M * N + thread_output_row_hbm * N + thread_output_col_hbm] = result;
    }
}

// A, B, C are device pointers
void solve(const float* const A,
           const float* const B,
           float* const C,
           int const BATCH,
           int const M,
           int const N,
           int const K) {
    dim3 const threadsPerBlock = dim3(SHM_TILE_SIZE, SHM_TILE_SIZE);
    dim3 const blocksPerGrid = dim3(ceil_div(N, threadsPerBlock.x),
                                    ceil_div(M, threadsPerBlock.y),
                                    BATCH);
    batched_matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, BATCH, M, N, K);
} 
