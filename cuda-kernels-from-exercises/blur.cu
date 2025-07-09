#DEFINE BLUR_SIZE 1

__global__ void blur(unsigned char* in, unsigned char* out, int const w, int const h) {
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int result = 0;
        int num_pixels = 0;
        for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
            int const current_x = x+i;
            for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                int const current_y = y+j;
                if (current_x >= 0 && current_x < w && current_y >= 0 && current_y < h) {
                    num_pixels++;
                    result += in[current_y*w + current_x];
                }
            }
        }
        out[y*w + i] = (unsigned char)(result / num_pixels);
    }
}
