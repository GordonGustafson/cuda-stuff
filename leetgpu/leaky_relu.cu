#include "solve.h"
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float const value = input[i];
        if (value > 0) {
            output[i] = value;
        } else {
            output[i] = 0.01 * value;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
