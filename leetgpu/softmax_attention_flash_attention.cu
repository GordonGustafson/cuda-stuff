#include "solve.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>

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
                                       float* const output_HBM,   // size Mxd
                                       int const M,
                                       int const N,
                                       int const d,
                                       float* const row_sum_HBM;
                                       float* const row_max_HBM;
                                       int const maxSharedMemory) {
    extern __shared__ float sharedMemory[];
    int const B_c = CEIL_DIV(maxSharedMemory, 4*d);
    int const B_r = std::min(B_c, d);
    int const T_c = CEIL_DIV(N, B_c);
    int const T_r = CEIL_DIV(M, B_r);

    float* const Q = sharedMemory;
    float* const K = Q + B_r * d;
    float* const V = K + B_c * d;
    float* const S = V + B_c * d;

    // Initialize S
    for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
        S[B_r_index * B_c + threadIdx.x] = 0.0f;
    }

    // Load Q
    for (int d_index = 0; d_index < d; d_index++) {
        if (threadIdx.x < B_r) {
            Q[threadIdx.x * d + d_index] = Q_HBM[blockIdx.x * d * B_r + threadIdx.x * d + d_index];
        }
    }

    // Iterate horizontally different S blocks.
    for (int T_c_index = 0; T_c_index < T_c; T_c_index++) {
        // Load K and V
        for (int d_index = 0; d_index < d; d_index++) {
            K[threadIdx.x * d + d_index] = K_HBM[T_c_index * d * B_c + threadIdx.x * d + d_index];
            V[threadIdx.x * d + d_index] = V_HBM[T_c_index * d * B_c + threadIdx.x * d + d_index];
        }

        __syncthreads();

        // Iterate vertically within the S block.
        for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
            float S_row_max = -INFINITY;
            float S_row_sum = 0.0f
            float result = 0.0f;
            for (int d_index = 0; d_index < d; d_index++) {
                result += Q[B_r_index * d + d_index] * K[threadIdx.x * d + d_index];
            }
            S[B_r_index * B_c + threadIdx.x] = result;
            __syncthreads();

            // Update max and sum for this row.
            if (threadIdx.x == 0) {
                for (int col = 0; col < B_c; col++) {
                    float const S_val = S[B_r_index * B_c + col];
                    S_row_sum = onlineSoftmaxSum(S_row_max, S_row_Sum, S_val, 1.0f);
                    S_row_max = std::max(S_row_max, S_val);
                }
                int const row_index = blockIdx * B_r + B_r_index;
                row_sum_HBM[row_index] = onlineSoftmaxSum(row_max_HBM[row_index],
                                                        row_sum_HBM[row_index],
                                                        S_row_max,
                                                        S_row_sum);
                row_max_HBM[row_index] = std::max(row_max_HBM[row_index], S_row_max);
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
    cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory);

    int const B_c = CEIL_DIV(maxSharedMemory, 4*d);
    int const B_r = std::min(B_c, d);
    int const T_c = CEIL_DIV(N, B_c);
    int const T_r = CEIL_DIV(M, B_r);


    float* zeroFloats = new float[M*d]();
    cudaMemcpy(output, zeroFloats, M * d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 const blocksPerGrid(T_r);
    dim3 const threadsPerBlock(B_c);
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, maxSharedMemory>>>(Q, K, V, output, M, N, d, maxSharedMemory);
}

