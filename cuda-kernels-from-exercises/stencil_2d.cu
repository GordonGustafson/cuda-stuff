#include <cuda_runtime.h>
#include <iostream>

int constexpr shmTileNumCols = 32;
int constexpr shmTileNumRows = 32;
int constexpr stencilSpokeLength = 1;
int constexpr outputTileNumCols = shmTileNumCols - 2 * stencilSpokeLength;
int constexpr outputTileNumRows = shmTileNumRows - 2 * stencilSpokeLength;


int constexpr ceilDiv(int dividend, int divisor) {
    return (dividend + divisor - 1) / divisor;
}

__global__ void stencil2DKernel(float const * const input,
                                int const inputNumCols,
                                int const inputNumRows,
                                float const upWeight,
                                float const downWeight,
                                float const leftWeight,
                                float const rightWeight,
                                float const centerWeight,
                                float* const output) {
    float __shared__ inputShm[shmTileNumRows][shmTileNumCols];

    int const shmTopRowHBM = blockIdx.y * outputTileNumRows - stencilSpokeLength;
    int const shmLeftColHBM = blockIdx.x * outputTileNumCols - stencilSpokeLength;
    int const shmRowHBM = shmTopRowHBM + threadIdx.y;
    int const shmColHBM = shmLeftColHBM + threadIdx.x;

    // Load input into inputShm

    if (shmRowHBM >= 0 && shmRowHBM < inputNumRows &&
        shmColHBM >= 0 && shmColHBM < inputNumCols) {
        inputShm[threadIdx.y][threadIdx.x] = input[shmRowHBM * inputNumCols + shmColHBM];
    } else {
        inputShm[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute and write result
    int const outputRowHBM = blockIdx.y * outputTileNumRows + threadIdx.y;
    int const outputColHBM = blockIdx.x * outputTileNumCols + threadIdx.x;
    if (threadIdx.x < outputTileNumCols && threadIdx.y < outputTileNumRows
        && outputRowHBM < inputNumRows && outputColHBM < inputNumCols) {
        int const inputRowShm = threadIdx.y + stencilSpokeLength;
        int const inputColShm = threadIdx.x + stencilSpokeLength;
        float result = (centerWeight * inputShm[inputRowShm][inputColShm]
                        + upWeight * inputShm[inputRowShm-1][inputColShm]
                        + downWeight * inputShm[inputRowShm+1][inputColShm]
                        + leftWeight * inputShm[inputRowShm][inputColShm-1]
                        + rightWeight * inputShm[inputRowShm][inputColShm+1]);
       
        output[outputRowHBM * inputNumCols + outputColHBM] = result;
    }
}


void stencil2D(float const * const input,
               int const inputNumCols,
               int const inputNumRows,
               float const upWeight,
               float const downWeight,
               float const leftWeight,
               float const rightWeight,
               float const centerWeight,
               float* const output) {
    dim3 const blocksPerGrid = dim3(ceilDiv(inputNumCols, outputTileNumCols),
                                    ceilDiv(inputNumRows, outputTileNumRows));
    dim3 const threadsPerBlock = dim3(shmTileNumCols, shmTileNumRows);
    stencil2DKernel<<<blocksPerGrid, threadsPerBlock>>>(input, inputNumCols, inputNumRows, upWeight, downWeight, leftWeight, rightWeight, centerWeight, output);
}


int main() {
    int const inputRowLength = 34;
    int const inputColLength = 34;
    float* inputHost = new float[inputRowLength * inputColLength]();
    for (int row = 0; row < inputRowLength; row++) {
        for (int col = 0; col < inputColLength; col++) {
            inputHost[row * inputColLength + col] = row * inputColLength + col;
        }
    }

    int const numBytes = inputRowLength * inputColLength * sizeof(float);
    float* inputDevice;
    float* outputDevice;
    cudaMalloc((void**)&inputDevice, numBytes);
    cudaMalloc((void**)&outputDevice, numBytes);
    cudaMemcpy(inputDevice, inputHost, numBytes, cudaMemcpyHostToDevice);

    stencil2D(inputDevice, inputColLength, inputRowLength, 1, 1, 1, 1, 1, outputDevice);

    float* outputHost = new float[inputRowLength * inputColLength]();
    cudaMemcpy(outputHost, outputDevice, numBytes, cudaMemcpyDeviceToHost);
        
    for (int row = 0; row < inputRowLength; row++) {
        for (int col = 0; col < inputColLength; col++) {
            std::cout << outputHost[row * inputColLength + col] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(inputDevice);
    cudaFree(outputDevice);

    delete[] inputHost;
    delete[] outputHost;

    return 0;
}
