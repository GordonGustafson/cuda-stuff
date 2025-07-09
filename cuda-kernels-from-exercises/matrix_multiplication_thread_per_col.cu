#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int const col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < K) {
        float result = 0.0f;
        for (int row = 0; row < M; row++) {
            for (int i = 0; i < N; i++) {
                result += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = result;
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 1);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, 1)
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
