#include <hip/hip_runtime.h>
#include <iostream>

#define ROWS 1024
#define COLUMNS 1024
#define THREADS_PER_BLOCK 256
#define BLOCK_DIM 16
#define SCALAR_T float

__global__ void mat_trans_kernel(float *d_in, float *d_out, size_t width, size_t height) {
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx < width && y_idx < height) {
        int inIdx = y_idx * width + x_idx;
        int outIdx = x_idx * height + y_idx;
        d_out[outIdx] = d_in[inIdx];
    }
}

__global__ void lds_mat_trans_kernel(float *d_in, float *d_out, size_t width, size_t height){
    __shared__ float shrd_mem[BLOCK_DIM][BLOCK_DIM];   // LDS

    int x_tile_idx = blockIdx.x * BLOCK_DIM;
    int y_tile_idx = blockIdx.y * BLOCK_DIM;

    int inIdx = (y_tile_idx + threadIdx.y) * width + (x_tile_idx + threadIdx.x);
    int outIdx = (x_tile_idx + threadIdx.x) * height + (y_tile_idx + threadIdx.y);

    shrd_mem[threadIdx.y][threadIdx.x] = d_in[inIdx];
    __syncthreads();
    d_out[outIdx] = shrd_mem[threadIdx.y][threadIdx.x];
}

int main()
{
    // Host memory allocation
    SCALAR_T *h_in = new SCALAR_T[ROWS * COLUMNS];
    SCALAR_T *h_out = new SCALAR_T[ROWS * COLUMNS];

    // Init host input array
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLUMNS; j++)
            h_in[i * COLUMNS + j] = static_cast<SCALAR_T>(i + j);

    // Declare device arrays
    SCALAR_T *d_in, *d_out;
    hipError_t err = hipMalloc(&d_in, COLUMNS * ROWS * sizeof(SCALAR_T));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed for d_in: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_out, COLUMNS * ROWS * sizeof(SCALAR_T));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed for d_out: " << hipGetErrorString(err) << std::endl;
        hipFree(d_in);
        return -1;
    }

    err = hipMemcpy(d_in, h_in, COLUMNS * ROWS * sizeof(SCALAR_T), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_in);
        hipFree(d_out);
        return -1;
    }
    
    // GPU Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((COLUMNS + blockDim.x - 1) / blockDim.x, (ROWS + blockDim.y - 1) / blockDim.y);
    // mat_trans_kernel<<<gridDim, blockDim>>>(d_in, d_out, COLUMNS, ROWS);
    lds_mat_trans_kernel<<<gridDim, blockDim>>>(d_in, d_out, COLUMNS, ROWS);
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_in);
        hipFree(d_out);
        return -1;
    }

    // Copy from device to host
    err = hipMemcpy(h_out, d_out, COLUMNS * ROWS * sizeof(SCALAR_T), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_in);
        hipFree(d_out);
        return -1;
    }


    // std::cout << "Result Comparison:\n" << std::endl;
    int num_diff = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            int out_idx = j * ROWS + i;
            if (h_in[i * COLUMNS + j] != h_out[out_idx]) {
                std::cerr << "Find an error." << std::endl;
                num_diff++;
            }
        }
    }
    std::cout << "Number of result differences: " << num_diff << ".\n";

    // Free device and host memory
    hipFree(d_in);
    hipFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
