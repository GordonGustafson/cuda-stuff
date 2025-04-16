#include "solve.h"
#include <cuda_runtime.h>

// 
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int const row = blockIdx.x * blockDim.x + threadIdx.x;
    int const col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < K) {
        float result = 0.0f;
        for (int i = 0; i < N; i++) {
            result += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = result;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
