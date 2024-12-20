#include <cuda_runtime.h>

extern "C" __global__ void process_image(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int batch_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;

    if (x < width && y < height && img_idx < batch_size) {
        int pixel_idx = img_idx * width * height * 3 + (y * width + x) * 3;

        // Copy the pixel values from input to output
        output[pixel_idx] = input[pixel_idx];       // Red
        output[pixel_idx + 1] = input[pixel_idx + 1]; // Green
        output[pixel_idx + 2] = input[pixel_idx + 2]; // Blue
    }
}
