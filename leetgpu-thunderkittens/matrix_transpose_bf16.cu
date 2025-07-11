#include "ThunderKittens/include/kittens.cuh"
#include <cuda_runtime.h>

using namespace kittens;

int constexpr REG_TILE_NUM_ROWS = 16;
int constexpr REG_TILE_NUM_COLS = 16;

int constexpr BLOCK_DIM_X = 32;

using _gl = gl<bf16, -1, -1, -1, -1, st_fl<REG_TILE_NUM_ROWS, REG_TILE_NUM_COLS>>;

struct transpose_globals {
    _gl input, output;
};

int constexpr ceil_div(int const dividend, int const divisor) {
    return (dividend + divisor - 1) / divisor;
}

__global__ void matrix_transpose_bf16_kernel(const __grid_constant__ transpose_globals globals) {
    // register memory 
    rt_bf<REG_TILE_NUM_ROWS, REG_TILE_NUM_COLS> input_reg;
    rt_bf<REG_TILE_NUM_ROWS, REG_TILE_NUM_COLS> output_reg;

    // load from HBM to registers
    load(input_reg, globals.input, {0, 0, (int)blockIdx.y, (int)blockIdx.x});
    __syncthreads();

    // Transpose register tile
    transpose_sep(output_reg, input_reg);
    __syncthreads();

    // store from registers to HBM
    store(globals.output, output_reg, {0, 0, (int)blockIdx.x, (int)blockIdx.y});
    __syncthreads();
}

void matrix_transpose_bf16(bf16 *d_input, bf16 *d_output, unsigned int rows, unsigned int cols) {
    _gl input_tile {d_input,  1, 1, rows, cols};
    _gl output_tile{d_output, 1, 1, cols, rows};
    transpose_globals globals{input_tile, output_tile};
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        matrix_transpose_bf16_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 const blocksPerGrid = dim3(ceil_div(cols, REG_TILE_NUM_COLS),
                                    ceil_div(rows, REG_TILE_NUM_ROWS));
    dim3 const threadsPerBlock = dim3(BLOCK_DIM_X);
    matrix_transpose_bf16_kernel<<<blocksPerGrid, threadsPerBlock, mem_size>>>(globals);
    cudaDeviceSynchronize();
}


// Utility to convert float to __nv_bfloat16
__nv_bfloat16 to_bf16(float f) {
    return __float2bfloat16(f);
}

// Utility to convert __nv_bfloat16 to float
float from_bf16(__nv_bfloat16 b) {
    return __bfloat162float(b);
}

void print_matrix(const float* data, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::cout << data[i * cols + j] << "\t";
        std::cout << "\n";
    }
}


void test_transpose_bf16(int rows, int cols) {
    size_t size = rows * cols * sizeof(__nv_bfloat16);

    // Allocate host input/output (float for easier validation)
    float* h_input_f = new float[rows * cols];
    float* h_output_f = new float[rows * cols];

    // Fill with known values
    for (int i = 0; i < rows * cols; ++i)
        // bfloat16 only has 7 mantissa bits, so 257 gets rounded to 256.
        // Capping the values at 256 avoids this problem, though at the cost
        // of non-ideal testing (we might not notice values getting mapped to
        // the wrong place since not every value is unique).
        h_input_f[i] = static_cast<float>(i % 256);

    // Convert to BF16
    __nv_bfloat16* h_input_bf16 = new __nv_bfloat16[rows * cols];
    for (int i = 0; i < rows * cols; ++i)
        h_input_bf16[i] = to_bf16(h_input_f[i]);

    // Device memory
    __nv_bfloat16 *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input_bf16, size, cudaMemcpyHostToDevice);

    // Launch kernel
    matrix_transpose_bf16(d_input, d_output, rows, cols);

    // Copy result back
    __nv_bfloat16* h_result_bf16 = new __nv_bfloat16[rows * cols];
    cudaMemcpy(h_result_bf16, d_output, size, cudaMemcpyDeviceToHost);

    // Convert output to float
    for (int i = 0; i < rows * cols; ++i)
        h_output_f[i] = from_bf16(h_result_bf16[i]);

    // Validation
    bool success = true;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            float expected = h_input_f[i * cols + j];
            float actual = h_output_f[j * rows + i];
            if (fabs(expected - actual) > 1e-2f) {
                std::cout << "Mismatch at (" << j << "," << i << ") in output: "
                          << "expected " << expected << ", got " << actual << "\n";
                success = false;
            }
        }

    if (success)
        std::cout << "✅ Transpose test passed for " << rows << "x" << cols << " matrix\n";
    else
        std::cout << "❌ Transpose test failed!\n";

    // Cleanup
    delete[] h_input_f;
    delete[] h_output_f;
    delete[] h_input_bf16;
    delete[] h_result_bf16;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    test_transpose_bf16(32, 32);    // square matrix
    test_transpose_bf16(16, 16);    // square matrix
    test_transpose_bf16(8, 8);      // square matrix
    test_transpose_bf16(4, 4);      // square matrix
    test_transpose_bf16(8, 5);      // rectangular (more cols)
    test_transpose_bf16(5, 8);      // rectangular (more rows)
    test_transpose_bf16(1, 1);      // edge case
    test_transpose_bf16(0, 0);      // empty matrix

    return 0;
}
