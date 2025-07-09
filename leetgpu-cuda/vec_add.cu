#include <stdio.h>

#define ARRAY_SIZE 8

__global__
void vec_add_cuda(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vec_add_cuda_stub(float* a_h, float* b_h, float* c_h, int n) {
      int size = n * sizeof(float); 
      float *a_d, *b_d, *c_d;

      // Part 1: Allocate device memory for a, b, and c
      cudaMalloc((void**)&a_d, size);
      cudaMalloc((void**)&b_d, size);
      cudaMalloc((void**)&c_d, size);

      // Copy a and b to device memory
      cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
      cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

      // Part 2: Call kernel – to launch a grid of threads
      // to perform the actual vector addition

      vec_add_cuda<<<ceil(n / 256.0), 256>>>(a_d, b_d, c_d, n);

      // Part 3: Copy c from the device memory
      cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

      // Free device vectors
      cudaFree(a_d);
      cudaFree(b_d);
      cudaFree(c_d);

}

void vec_print_host(float* a_h, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f\n", a_h[i]);
    }
}

int main(int argc, char** argv) {

    float a[ARRAY_SIZE] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float b[ARRAY_SIZE] = {0.0f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
    float c[ARRAY_SIZE];

    vec_add_cuda_stub(a, b, c, ARRAY_SIZE);

    vec_print_host(c, ARRAY_SIZE);
    
    return 0;
}


