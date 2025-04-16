#include <cuda_runtime.h>

#define OUTPUT_TILE_WIDTH 30
#define INPUT_TILE_WIDTH (OUTPUT_TILE_WIDTH + 2)
#define CEIL_DIV(dividend, divisor) ((dividend + divisor - 1) / divisor)


// Output will be the same size as the input.
// Elements from the edges (boundary conditions) are copied from the input
__global__ void stencil_3d_thread_coarsening_kernel(float const* const in,
                                                    float* const out,
                                                    unsigned int const input_depth,
                                                    unsigned int const input_rows,
                                                    unsigned int const input_cols,
                                                    unsigned int const coarsening_factor,
                                                    float const c0,
                                                    float const c1,
                                                    float const c2,
                                                    float const c3,
                                                    float const c4,
                                                    float const c5,
                                                    float const c6) {
    float input_prev = 0.0f;
    __shared__ float input_curr[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
    float input_next = 0.0f;

    int const x = blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x - 1;
    int const y = blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y - 1;
    int const z = blockIdx.z * coarsening_factor;

    bool const x_in_bounds = (x >= 0 && x < input_cols);
    bool const y_in_bounds = (y >= 0 && y < input_rows);
    bool const z_in_bounds = (z >= 0 && z < input_depth);

    bool const is_output_thread = (threadIdx.x > 0 && threadIdx.x < INPUT_TILE_WIDTH - 1
                                   && threadIdx.y > 0 && threadIdx.y < INPUT_TILE_WIDTH - 1);

    if (x_in_bounds && y_in_bounds && z > 0 && z < input_depth) {
        input_prev = in[(z - 1) * input_rows * input_cols
                        + y * input_cols
                        + x];
    }
    if (x_in_bounds && y_in_bounds && z_in_bounds) {
        input_curr[threadIdx.y][threadIdx.x] = in[z * input_rows * input_cols
                                                  + y * input_cols
                                                  + x];
    }
    __syncthreads();

    for (int z_offset = 0; z + z_offset < input_depth && z_offset < coarsening_factor; z_offset++) {
        int const zloop = z + z_offset;
        bool const zloop_in_bounds = (zloop >= 0 && zloop < input_depth);
        if (x_in_bounds && y_in_bounds && zloop + 1 < input_depth) {
            input_next = in[(zloop + 1) * input_rows * input_cols
                            + y * input_cols
                            + x];
        }

        bool const is_internal_node = (x > 0 && x < input_cols - 1
                                       && y > 0 && y < input_rows - 1
                                       && zloop > 0 && zloop < input_depth - 1);
        bool const is_edge_node = (((x == 0 || x == input_cols - 1) && y_in_bounds && zloop_in_bounds)
                                   || ((y == 0 || y == input_rows - 1) && x_in_bounds && zloop_in_bounds)
                                   || ((zloop == 0 || zloop == input_depth - 1) && x_in_bounds && y_in_bounds));
        if (is_internal_node && is_output_thread) {
            float const result = (  c0 * input_curr[threadIdx.y][threadIdx.x]
                                  + c1 * input_curr[threadIdx.y][threadIdx.x-1]
                                  + c2 * input_curr[threadIdx.y][threadIdx.x+1]
                                  + c3 * input_curr[threadIdx.y-1][threadIdx.x]
                                  + c4 * input_curr[threadIdx.y+1][threadIdx.x]
                                  + c5 * input_prev
                                  + c6 * input_next);
            out[zloop * input_rows * input_cols
                + y * input_cols
                + x] = result;
        }
        // If it's on the boundary, copy the input into the output.
        if (is_edge_node && is_output_thread) {
            int const offset = (zloop * input_rows * input_cols
                                + y * input_cols
                                + x);
            out[offset] = input_curr[threadIdx.y][threadIdx.x];
        }
        __syncthreads();

        if (x_in_bounds && y_in_bounds) {
            input_prev = input_curr[threadIdx.y][threadIdx.x];
            input_curr[threadIdx.y][threadIdx.x] = input_next;
        }

        __syncthreads();
    }
}


void stencil_3d_thread_coarsening(float const* const in,
                                  float* const out,
                                  unsigned int const input_depth,
                                  unsigned int const input_rows,
                                  unsigned int const input_cols,
                                  unsigned int const coarsening_factor,
                                  float const c0,
                                  float const c1,
                                  float const c2,
                                  float const c3,
                                  float const c4,
                                  float const c5,
                                  float const c6) {
    dim3 const threadsPerBlock(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, 1);
    dim3 const blocksPerGrid(CEIL_DIV(input_cols, OUTPUT_TILE_WIDTH),
                             CEIL_DIV(input_rows, OUTPUT_TILE_WIDTH),
                             CEIL_DIV(input_depth, coarsening_factor));
    stencil_3d_thread_coarsening_kernel<<<blocksPerGrid, threadsPerBlock>>>(in, out, input_depth, input_rows, input_cols,
                                                                            coarsening_factor, c0, c1, c2, c3, c4, c5, c6);
}

