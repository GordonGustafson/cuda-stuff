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

int getMaxSharedMemPerBlockOptin() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.sharedMemPerBlockOptin;
}

__global__ void flash_attention_kernel(float const* const Q_HBM,  // size Mxd
                                       float const* const K_HBM,  // size Nxd
                                       float const* const V_HBM,  // size Nxd
                                       float* const output_HBM,   // size Mxd
                                       int const M,
                                       int const N,
                                       int const d,
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

    float l_i = 0.0f;
    float m_i = -INFINITY;

    for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
        S[B_r_index * d + threadIdx.x] = 0.0f;
    }

    // Load V and K.
    for (int B_c_index = 0; B_c_index < B_c; B_c_index++) {
        V[B_c_index * d + threadIdx.x] = V_HBM[blockIdx.x * d * B_c + B_c_index * d + threadIdx.x];
        K[B_c_index * d + threadIdx.x] = K_HBM[blockIdx.x * d * B_c + B_c_index * d + threadIdx.x];
    }

    // Load Q.
    for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
        Q[B_r_index * d + threadIdx.x] = V_HBM[blockIdx.y * d * B_r + B_r_index * d + threadIdx.x];
    }

    for (int B_c_index = 0; B_c_index < B_c; B_c_index++) {
        for (int B_r_index = 0; B_r_index < B_c; B_r_index++) {
            float result = 0.0f;
            for (int d_index = 0; d_index < d; d_index++) {
                result += Q[B_r_index * d + d_index] * K[B_r_index * d + d_index];
            }
            S[B_r_index * B_c + B_c_index] = result;
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
    int const maxSharedMemory = getMaxSharedMemPerBlockOptin();
    cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory);

    int const B_c = CEIL_DIV(maxSharedMemory, 4*d);
    int const B_r = std::min(B_c, d);
    int const T_c = CEIL_DIV(N, B_c);
    int const T_r = CEIL_DIV(M, B_r);


    float* zeroFloats = new float[M*d]();
    cudaMemcpy(output, zeroFloats, M * d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 const blocksPerGrid(T_c, T_r);
    dim3 const threadsPerBlock(B_c, B_r);
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, maxSharedMemory>>>(Q, K, V, output, M, N, d, maxSharedMemory);
}

