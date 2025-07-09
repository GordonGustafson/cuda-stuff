#DEFINE TILE_WIDTH 16

/* __syncthreads() */

__global__ void matrix_multiplication(float const * const M,
                                      float const * const N,
                                      float * const P,
                                      int const width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int const outputCol = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int const outputRow = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float result = 0;
    for (int tileIndex = 0; tileIndex < width / TILE_WIDTH; tileIndex++) {
        // Load data
        Mds[threadIndex.y][threadIdx.x] = M[outputRow * width + tileIndex * TILE_WIDTH + threadIdx.x];
        Nds[threadIndex.y][threadIdx.x] = N[(tileIndex * TILE_WIDTH + threadIdx.y) * width + outputCol];
        __syncthreads();

        for (int offsetInTile = 0; offsetInTile < TILE_WIDTH; offsetInTile++) {
            result += Mds[threadIdx.y][offsetInTile] * Nds[offsetInTile][threadIdx.x];
        }
        __syncthreads();
    }
    P[outputRow * width + outputCol] = result;
}
