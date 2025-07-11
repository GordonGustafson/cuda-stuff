#include "ThunderKittens/include/kittens.cuh"
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

using namespace kittens;

int constexpr SHM_NUM_ROWS = 16;
int constexpr SHM_NUM_COLS = 16;

using _gl = gl<float, -1, -1, -1, -1, st_fl<SHM_NUM_ROWS, SHM_NUM_COLS>>;

struct transpose_globals {
    _gl input, output;
};

__global__ __launch_bounds__(16, 16)
void matrix_transpose_kernel(const __grid_constant__ transpose_globals globals) {
    // register memory 
    rt_fl<SHM_NUM_ROWS, SHM_NUM_COLS> input_reg;
    rt_fl<SHM_NUM_ROWS, SHM_NUM_COLS> output_reg;

    // load from HBM to registers
    load(input_reg, globals.input, {0, 0, 0, 0});
    __syncthreads();

    // Transpose register tile
    transpose_sep(output_reg, input_reg);
    __syncthreads();

    // store from registers to HBM
    store(globals.output, output_reg, {0, 0, 0, 0});
    __syncthreads();
}

void matrix_transpose( float *d_input, float *d_output ) {
    _gl input_tile {d_input,  1, 1, SHM_NUM_ROWS, SHM_NUM_COLS};
    _gl output_tile{d_output, 1, 1, SHM_NUM_ROWS, SHM_NUM_COLS};
    transpose_globals globals{input_tile, output_tile};
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        matrix_transpose_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    matrix_transpose_kernel<<<16, 16, mem_size>>>(globals);
    cudaDeviceSynchronize();
}


// CPU-side reference transpose
void transposeCPU(const std::vector<float>& input, std::vector<float>& output, int width, int height) {
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            output[col * height + row] = input[row * width + col];
}

void testTranspose(int width, int height) {
    int size = width * height;
    std::vector<float> h_input(size);
    std::vector<float> h_output(size, -1.0f);        // result from GPU
    std::vector<float> h_expected(size, -2.0f);      // reference result

    // Fill input matrix with sequential values
    for (int i = 0; i < size; ++i)
        h_input[i] = static_cast<float>(i);

    float *d_input, *d_output;
    CUDACHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDACHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    matrix_transpose(d_input, d_output);
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    transposeCPU(h_input, h_expected, width, height);

    for (int i = 0; i < size; ++i) {
        if (fabs(h_output[i] - h_expected[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": GPU " << h_output[i] << " vs CPU " << h_expected[i] << std::endl;
            exit(1);
        }
    }

    std::cout << "Transpose test passed for " << height << "x" << width << " matrix.\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    testTranspose(4, 4);      // square matrix
    testTranspose(8, 5);      // rectangular (more cols)
    testTranspose(5, 8);      // rectangular (more rows)
    testTranspose(1, 1);      // edge case
    testTranspose(0, 0);      // empty matrix

    return 0;
}
