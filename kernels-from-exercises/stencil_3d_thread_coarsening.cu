#include <cuda_runtime.h>

#define OUTPUT_TILE_WIDTH 6
#define INPUT_TILE_WIDTH (OUTPUT_TILE_WIDTH + 2)
#define COARSENING_FACTOR 8
#define CEIL_DIV(dividend, divisor) ((dividend + divisor - 1) / divisor)

struct stencil_values {
    float const c0;
    float const c1;
    float const c2;
    float const c3;
    float const c4;
    float const c5;
    float const c6;
};

// Output will be the same size as the input.
// Elements from the edges (boundary conditions) are copied from the input
__global__ void stencil_3d_kernel(float const* const in,
                                  float* const out,
                                  unsigned int const input_depth,
                                  unsigned int const input_rows,
                                  unsigned int const input_cols,
                                  stencil_values const stencil) {
    __shared__ float input_prev[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
    __shared__ float input_curr[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
    __shared__ float input_next[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];

    int const x = blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x - 1;
    int const y = blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y - 1;
    int const z = (blockIdx.z * OUTPUT_TILE_WIDTH + threadIdx.z) * COARSENING_FACTOR - 1;

    bool const xy_in_bounds = (x > 0 && x < input_cols && y > 0 && y < input_rows);

    if (xy_in_bounds && z > 0 && z < input_depth) {
        input_prev[y][x] = in[z * input_rows * input_cols
                              + y * input_cols
                              + x];
    }
    if (xy_in_bounds && z > 0 && z + 1 < input_depth) {
        input_curr[y][x] = in[(z + 1) * input_rows * input_cols
                              + y * input_cols
                              + x];
    }

    for (z_offset = 0; z + z_offset < input_depth && z_offset < COARSENING_FACTOR; z_offset++) {
        if (xy_in_bounds && z + z_offset + 1 < input_depth) {
            input_next[y][x] = in[(z + z_offset + 1) * input_rows * input_cols
                                + y * input_cols
                                + x];
        }
        __syncthreads();

        bool const is_internal_node = (x > 0 && x < input_cols - 1
                                       && y > 0 && y < input_rows - 1
                                       && z + z_offset > 0 && z + z_offset < input_depth - 1);
        bool const is_edge_node = (x == 0 || x == input_cols - 1
                                   || y == 0 || y == input_rows - 1
                                   || z + z_offset == 0 || z + z_offset == input_depth - 1);
        if (is_internal_node) {
            float const result = (stencil.c0 * input_curr[y][x]
                                  + stencil.c1 * input_curr[y][x-1]
                                  + stencil.c2 * input_curr[y][x+1]
                                  + stencil.c3 * input_curr[y-1][x]
                                  + stencil.c4 * input_curr[y+1][x]
                                  + stencil.c5 * input_prev[y][x]
                                  + stencil.c6 * input_next[y][x]);
            out[(z+z_offset) * input_rows * input_cols
                + y * input_cols
                + x] = result;
        }
        // If it's on the boundary, copy the input into the output.
        if (is_edge_node) {
            int const offset = ((z+z_offset) * input_rows * input_cols
                                + y * input_cols
                                + x);
            // TODO: optimize this to copy from shared memory
            out[offset] = input[offset];
        }
        __syncthreads();

        if (xy_in_bounds) {
            input_prev[y][x] = input_curr[y][x];
            input_curr[y][x] = input_next[y][x];
        }
    }
}

void stencil_3d(float const* const in,
                float* const out,
                unsigned int const input_depth,
                unsigned int const input_rows,
                unsigned int const input_cols,
                stencil_values const stencil) {
    dim3 const threadsPerBlock(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, INPUT_TILE_WIDTH);
    dim3 const blocksPerGrid(CEIL_DIV(input_cols, OUTPUT_TILE_WIDTH),
                             CEIL_DIV(input_rows, OUTPUT_TILE_WIDTH),
                             CEIL_DIV(input_depth, (OUTPUT_TILE_WIDTH * COARSENING_FACTOR));
    stencil_3d_kernel<<<blocksPerGrid, threadsPerBlock>>>(in, out, input_depth, input_rows, input_cols, stencil);
}


