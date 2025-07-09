#define TILE_WIDTH 16

__global__ void matrix_multiplication_kernel(float const * const M,
                                             float const * const N,
                                             float * const P,
                                             int const M_dim,
                                             int const k,
                                             int const N_dim) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int const outputCol = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int const outputRow = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float result = 0;
    for (int tileIndex = 0; tileIndex < (k + TILE_WIDTH - 1) / TILE_WIDTH; tileIndex++) {
        if (outputRow < M_dim) {
            Mds[threadIdx.y][threadIdx.x] = M[outputRow * k + tileIndex * TILE_WIDTH + threadIdx.x];
        } else {
            Mds[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (outputCol < N_dim) {
            Nds[threadIdx.y][threadIdx.x] = N[(tileIndex * TILE_WIDTH + threadIdx.y) * N_dim + outputCol];
        } else {
            Nds[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int offsetInTile = 0; offsetInTile < TILE_WIDTH; offsetInTile++) {
            result += Mds[threadIdx.y][offsetInTile] * Nds[offsetInTile][threadIdx.x];
        }
        __syncthreads();
    }
    if (outputRow < M_dim && outputCol < N_dim) {
        P[outputRow * N_dim + outputCol] = result;
    }
}
