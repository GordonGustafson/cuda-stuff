#include <cuda_runtime.h>

#define OUTPUT_TILE_WIDTH 6
#define INPUT_TILE_WIDTH (OUTPUT_TILE_WIDTH + 2)
#define CEIL_DIV(dividend, divisor) ((dividend + divisor - 1) / divisor)

// Output will be the same size as the input.
// Elements from the edges (boundary conditions) are copied from the input
__global__ void stencil_3d_kernel(float const* const in,
                                  float* const out,
                                  unsigned int const input_depth,
                                  unsigned int const input_rows,
                                  unsigned int const input_cols,
                                  float const c0,
                                  float const c1,
                                  float const c2,
                                  float const c3,
                                  float const c4,
                                  float const c5,
                                  float const c6) {
    __shared__ float input_shared[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];

    int const x = blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x - 1;
    int const y = blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y - 1;
    int const z = blockIdx.z * OUTPUT_TILE_WIDTH + threadIdx.z - 1;

    bool const x_in_bounds = (x >= 0 && x < input_cols);
    bool const y_in_bounds = (y >= 0 && y < input_rows);
    bool const z_in_bounds = (z >= 0 && z < input_depth);

    if (x_in_bounds && y_in_bounds && z_in_bounds) {
        input_shared[threadIdx.z][threadIdx.y][threadIdx.x] = in[z * input_rows * input_cols
                                                                 + y * input_cols
                                                                 + x];
    }
    __syncthreads();

    bool const is_internal_node = (x > 0 && x < input_cols - 1
                                   && y > 0 && y < input_rows - 1
                                   && z > 0 && z < input_depth - 1
                                   && threadIdx.x > 0 && threadIdx.x <= OUTPUT_TILE_WIDTH
                                   && threadIdx.y > 0 && threadIdx.y <= OUTPUT_TILE_WIDTH
                                   && threadIdx.z > 0 && threadIdx.z <= OUTPUT_TILE_WIDTH);
    bool const is_edge_node = (((x == 0 || x == input_cols - 1) && y_in_bounds && z_in_bounds)
                               || ((y == 0 || y == input_rows - 1) && x_in_bounds && z_in_bounds)
                               || ((z == 0 || z == input_depth - 1) && x_in_bounds && y_in_bounds));
    if (is_internal_node) {
        float const result = (  c0 * input_shared[threadIdx.z][threadIdx.y][threadIdx.x]
                              + c1 * input_shared[threadIdx.z][threadIdx.y][threadIdx.x-1]
                              + c2 * input_shared[threadIdx.z][threadIdx.y][threadIdx.x+1]
                              + c3 * input_shared[threadIdx.z][threadIdx.y-1][threadIdx.x]
                              + c4 * input_shared[threadIdx.z][threadIdx.y+1][threadIdx.x]
                              + c5 * input_shared[threadIdx.z-1][threadIdx.y][threadIdx.x]
                              + c6 * input_shared[threadIdx.z+1][threadIdx.y][threadIdx.x]);
        out[z * input_rows * input_cols
            + y * input_cols
            + x] = result;
    }
    // If it's on the boundary, copy the input into the output.
    if (is_edge_node) {
        int const offset = (z * input_rows * input_cols
                            + y * input_cols
                            + x);
        out[offset] = input_shared[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void stencil_3d(float const* const in,
                float* const out,
                unsigned int const input_depth,
                unsigned int const input_rows,
                unsigned int const input_cols,
                float const c0,
                float const c1,
                float const c2,
                float const c3,
                float const c4,
                float const c5,
                float const c6) {
    dim3 const threadsPerBlock(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, INPUT_TILE_WIDTH);
    dim3 const blocksPerGrid(CEIL_DIV(input_cols, OUTPUT_TILE_WIDTH),
                             CEIL_DIV(input_rows, OUTPUT_TILE_WIDTH),
                             CEIL_DIV(input_depth, OUTPUT_TILE_WIDTH));
    stencil_3d_kernel<<<blocksPerGrid, threadsPerBlock>>>(in, out, input_depth, input_rows, input_cols,
                                                          c0, c1, c2, c3, c4, c5, c6);
}


