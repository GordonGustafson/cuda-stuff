#include "solve.h"
#include <cuda_runtime.h>
#include <limits>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

__device__ static inline float onlineSoftmaxSum(float const maxA,
                                                float const sumA,
                                                float const maxB,
                                                float const sumB) {
    if (sumA == 0.0f) {
        return sumB;
    } else if (sumB == 0.0f) {
        return sumA;
    } else if (maxA > maxB) {
        return sumB * expf(maxB - maxA) + sumA;
    } else {
        return sumB + sumA * expf(maxA - maxB);
    }
}

__global__ void flash_attention_kernel(float const* const Q_HBM,  // size Mxd
                                       float const* const K_HBM,  // size Nxd
                                       float const* const V_HBM,  // size Nxd
                                       float* const O_HBM,        // size Mxd
                                       int const M,
                                       int const N,
                                       int const d,
                                       float const temperature,
                                       float* const row_sum_HBM,
                                       float* const row_max_HBM,
                                       int const maxSharedMemory) {
    extern __shared__ float sharedMemory[];
    int const B_c = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)N);
    int const B_r = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)d);
    int const T_c = CEIL_DIV(N, B_c);

    int const B_r_bounds_checked_for_last_row = min(B_r, M - blockIdx.x * B_r);

    float* const Q = sharedMemory;
    float* const K = Q + B_r * d;
    float* const V = K + B_c * d;
    float* const S = V + B_c * d;

    // Initialize S, using threadIdx.x as the B_c dimension.
    for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
        S[B_r_index * B_c + threadIdx.x] = 0.0f;
    }

    // Load Q, using threadIdx.x to help along the d dimension
    for (int d_index = threadIdx.x; d_index < d; d_index += blockDim.x) {
        for (int B_r_index = 0; B_r_index < B_r_bounds_checked_for_last_row; B_r_index++) {
            int const row_index = blockIdx.x * B_r + B_r_index;
            Q[B_r_index * d + d_index] = Q_HBM[row_index * d + d_index];
        }
    }

    // Iterate horizontally through different S blocks.
    for (int T_c_index = 0; T_c_index < T_c; T_c_index++) {
        int const B_c_bounds_checked_for_last_column = min(B_c, N - T_c_index * B_c);
        // Load K and V
        for (int d_index = threadIdx.x; d_index < d; d_index += blockDim.x) {
            for (int B_c_index = 0; B_c_index < B_c_bounds_checked_for_last_column; B_c_index++) {
                int const row_index = T_c_index * B_c + B_c_index;
                K[B_c_index * d + d_index] = K_HBM[row_index * d + d_index];
                V[B_c_index * d + d_index] = V_HBM[row_index * d + d_index];
            }
        }

        __syncthreads();

        // Iterate vertically within the S block.
        for (int B_r_index = 0; B_r_index < B_r_bounds_checked_for_last_row; B_r_index++) {
            float S_val_for_thread = 0.0f;
            for (int d_index = 0; d_index < d; d_index++) {
                S_val_for_thread += Q[B_r_index * d + d_index] * K[threadIdx.x * d + d_index];
            }
            S[B_r_index * B_c + threadIdx.x] = S_val_for_thread / temperature;

            int const row_index = blockIdx.x * B_r + B_r_index;
            float const S_row_old_global_max = row_max_HBM[row_index];
            float const S_row_old_global_sum = row_sum_HBM[row_index];
            __syncthreads();

            // Update max and sum for this row.
            if (threadIdx.x == 0) {
                float S_row_local_max = -INFINITY;
                float S_row_local_sum = 0.0f;
                for (int col = 0; col < B_c_bounds_checked_for_last_column; col++) {
                    float const S_val_iter = S[B_r_index * B_c + col];
                    S_row_local_sum = onlineSoftmaxSum(S_row_local_max, S_row_local_sum, S_val_iter, 1.0f);
                    S_row_local_max = max(S_row_local_max, S_val_iter);
                }
                row_sum_HBM[row_index] = onlineSoftmaxSum(S_row_old_global_max,
                                                            S_row_old_global_sum,
                                                            S_row_local_max,
                                                            S_row_local_sum);
                row_max_HBM[row_index] = max(S_row_old_global_max, S_row_local_max);
            }
            __syncthreads();
            float const S_row_new_global_max = row_max_HBM[row_index];
            float const S_row_new_global_sum = row_sum_HBM[row_index];

            // Compute P and O
            for (int d_index = threadIdx.x; d_index < d; d_index += blockDim.x) {
                float PV_val = 0.0f;
                for (int V_B_c_index = 0; V_B_c_index < B_c_bounds_checked_for_last_column; V_B_c_index++) {
                    float const S_val = S[B_r_index * B_c + V_B_c_index];
                    float const P_val = expf(S_val - S_row_new_global_max) / S_row_new_global_sum;
                    PV_val += P_val * V[V_B_c_index * d + d_index];
                }

                int const row_index = blockIdx.x * B_r + B_r_index;
                int const OIndexForThread = row_index * d + d_index;
                O_HBM[OIndexForThread] = O_HBM[OIndexForThread] * expf(S_row_old_global_max - S_row_new_global_max) * (S_row_old_global_sum / S_row_new_global_sum) + PV_val;
            }
        }
    }
}


// Q, K, V, output are device pointers
void solve(float const* const Q,  // size Mxd
           float const* const K,  // size Nxd
           float const* const V,  // size Nxd
           float* const output,   // size Mxd
           int const M,
           int const N,
           int const d) {
    int maxSharedMemory;
    gpuErrchk(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    gpuErrchk(cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));

    int const B_c = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)N);
    int const B_r = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)d);
    int const T_r = CEIL_DIV(M, B_r);

    std::cout << "maxSharedMemory: " << maxSharedMemory << std::endl;
    std::cout << "B_c: " << B_c << std::endl;
    std::cout << "B_r: " << B_r << std::endl;
    std::cout << "T_r: " << T_r << std::endl;

    float* row_sum_HBM;
    gpuErrchk(cudaMalloc((void**)&row_sum_HBM, M * sizeof(float)));
    float* row_max_HBM;
    gpuErrchk(cudaMalloc((void**)&row_max_HBM, M * sizeof(float)));

    float* zeroFloats = new float[M*d]();
    gpuErrchk(cudaMemcpy(output, zeroFloats, M * d * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(row_sum_HBM, zeroFloats, M * sizeof(float), cudaMemcpyHostToDevice));

    float* negativeInfinityFloats = new float[M];
    std::fill(negativeInfinityFloats, negativeInfinityFloats + M, -INFINITY);
    gpuErrchk(cudaMemcpy(row_max_HBM, negativeInfinityFloats, M * sizeof(float), cudaMemcpyHostToDevice));

    float const temperature = sqrt(d);

    dim3 const blocksPerGrid(T_r);
    dim3 const threadsPerBlock(B_c);
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, maxSharedMemory>>>(Q, K, V, output, M, N, d, temperature, row_sum_HBM, row_max_HBM, maxSharedMemory);
    gpuErrchk(cudaPeekAtLastError());

    delete[] zeroFloats;
    delete[] negativeInfinityFloats;
}



// Host utility to print Mxd matrix
void print_matrix(const float* mat, int M, int d) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j)
            std::cout << mat[i * d + j] << " ";
        std::cout << std::endl;
    }
}

int main() {
    const int M = 2000, N = 3000, d = 1024;

    // Input data
    float* const Q_host = new float[M*d]();
    float* const K_host = new float[N*d]();
    float* const V_host = new float[N*d]();

    const size_t size_Q = M * d * sizeof(float);
    const size_t size_KV = N * d * sizeof(float);
    const size_t size_O = size_Q;

    float* Q_dev;
    float* K_dev;
    float* V_dev;
    float* O_dev;

    cudaMalloc(&Q_dev, size_Q);
    cudaMalloc(&K_dev, size_KV);
    cudaMalloc(&V_dev, size_KV);
    cudaMalloc(&O_dev, size_O);

    cudaMemcpy(Q_dev, Q_host, size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(K_dev, K_host, size_KV, cudaMemcpyHostToDevice);
    cudaMemcpy(V_dev, V_host, size_KV, cudaMemcpyHostToDevice);

    solve(Q_dev, K_dev, V_dev, O_dev, M, N, d);
    cudaDeviceSynchronize();

    // Copy result back
    float* const O_host = new float[M * d]();
    cudaMemcpy(O_host, O_dev, size_O, cudaMemcpyDeviceToHost);

    // Print output
    std::cout << "Output O:" << std::endl;
    print_matrix(O_host, M, d);

    // Free memory
    cudaFree(Q_dev);
    cudaFree(K_dev);
    cudaFree(V_dev);
    cudaFree(O_dev);

    return 0;
}


