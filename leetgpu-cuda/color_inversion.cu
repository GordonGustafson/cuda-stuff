#include "solve.h"
#include <cuda_runtime.h>

/*
Write a CUDA program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer (unsigned char).

Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

The input array image will contain width * height * 4 elements. The first 4 elements represent the RGBA values of the top-left pixel, the next 4 elements represent the pixel to its right, and so on.
*/

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        int const pixelOffset = i * 4;
        image[pixelOffset]   = 255 - image[pixelOffset];
        image[pixelOffset+1] = 255 - image[pixelOffset+1];
        image[pixelOffset+2] = 255 - image[pixelOffset+2];
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
