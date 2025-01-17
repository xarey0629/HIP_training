#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <vector>
using namespace std;

#define MATRIX_SIZE 1024
#define SCALAR_T float
#define THREADS_PER_2D_BLOCK 32

#define ROW_MAJOR 0

/*
    HIP implementation of matrix multiplication
*/

// row-major matrix multiplication on GPU
template <typename T>
__global__ void k_matrix_multiply_row_major(T* a, T* b, T* out, size_t dim) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;  // Column
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // Row
    if (x < dim  && y < dim) {
        T sum = 0;
        for (int k = 0; k < dim; k++)
            sum += a[y * dim + k] * b[k * dim + x];
        out[y * dim + x] = sum;
    }
}

// column-major matrix multiplication on GPU
template <typename T>
__global__ void k_matrix_multiply_column_major(T* a, T* b, T* out, size_t dim) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;  // Column
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // Row
    if (x < dim  && y < dim) {
        T sum = 0;
        for (int k = 0; k < dim; k++)
            sum += a[k * dim + x] * b[y * dim + k];
        out[y * dim + x] = sum;
    }
}

// tile-based row-major matrix multiplication on GPU
template <typename T>
__global__ void k_tiled_matrix_multiply_row_major(T* a, T* b, T* out, size_t dim) {
    __shared__ T a_tile[THREADS_PER_2D_BLOCK][THREADS_PER_2D_BLOCK];
    __shared__ T b_tile[THREADS_PER_2D_BLOCK][THREADS_PER_2D_BLOCK];

    int x = blockDim.x * blockIdx.x + threadIdx.x;  // Column
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // Row

    T out_k = 0;
    for (int i = 0; i < dim / THREADS_PER_2D_BLOCK; i++) {
        a_tile[threadIdx.y][threadIdx.x] = a[y * dim + i * THREADS_PER_2D_BLOCK + threadIdx.x];
        b_tile[threadIdx.y][threadIdx.x] = b[(i * THREADS_PER_2D_BLOCK + threadIdx.y) * dim + x];
        __syncthreads();

        for (int k = 0; k < THREADS_PER_2D_BLOCK; k++)
            out_k += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        __syncthreads();
    }
    out[y * dim + x] = out_k;
}

// tile-based column-major matrix multiplication on GPU
// TODO: Validation of the results
template <typename T>
__global__ void k_tiled_matrix_multiply_column_major(T* a, T* b, T* out, size_t dim) {
    __shared__ T a_tile[THREADS_PER_2D_BLOCK][THREADS_PER_2D_BLOCK];
    __shared__ T b_tile[THREADS_PER_2D_BLOCK][THREADS_PER_2D_BLOCK];

    int x = blockDim.x * blockIdx.x + threadIdx.x;  // Column
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // Row

    T out_k = 0;
    for (int i = 0; i < dim / THREADS_PER_2D_BLOCK; i++) {
        a_tile[threadIdx.y][threadIdx.x] = a[(i * THREADS_PER_2D_BLOCK + threadIdx.y) * dim + x];
        b_tile[threadIdx.y][threadIdx.x] = b[y * dim + i * THREADS_PER_2D_BLOCK + threadIdx.x];
        __syncthreads();

        for (int k = 0; k < THREADS_PER_2D_BLOCK; k++)
            out_k += a_tile[k][threadIdx.x] * b_tile[threadIdx.y][k];
        __syncthreads();
    }
    out[y * dim + x] = out_k;
}


/*
    CPU implementation of matrix multiplication
*/

// row-major matrix multiplication on CPU
template <typename T>
void matrix_multiply_row_major_cpu(T* a, T* b, T* out, size_t dim) {
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            T sum = 0;
            for(int k = 0; k < dim; k++) {
                sum += a[i * dim + k] * b[k * dim + j];
            }
            out[i * dim + j] = sum;
        }
    }
}

// column-major matrix multiplication on CPU
template <typename T>
void matrix_multiply_column_major_cpu(T* a, T* b, T* out, size_t dim) {
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            T sum = 0;
            for(int k = 0; k < dim; k++) {
                sum += a[k * dim + i] * b[j * dim + k];
            }
            out[j * dim + i] = sum;
        }
    }
}

int main()
{
    // vector<SCALAR_T> arrayA(MATRIX_SIZE * MATRIX_SIZE);      // Matrix A
    // vector<SCALAR_T> arrayB(MATRIX_SIZE * MATRIX_SIZE);      // Matrix B
    // vector<SCALAR_T> arrayC(MATRIX_SIZE * MATRIX_SIZE, 0);   // Matrix C (for CPU)
    // vector<SCALAR_T> arrayC_gpu(MATRIX_SIZE * MATRIX_SIZE);  // Matrix C (for GPU returns)
    SCALAR_T* arrayA = new SCALAR_T[MATRIX_SIZE * MATRIX_SIZE];
    SCALAR_T* arrayB = new SCALAR_T[MATRIX_SIZE * MATRIX_SIZE];
    SCALAR_T* arrayC = new SCALAR_T[MATRIX_SIZE * MATRIX_SIZE];
    SCALAR_T* arrayC_gpu = new SCALAR_T[MATRIX_SIZE * MATRIX_SIZE];

    SCALAR_T* d_arrayA; // Matrix Array A on device
    SCALAR_T* d_arrayB; // Matrix Array B on device
    SCALAR_T* d_arrayC; // Result Array on device

    int i, j, k;

    // Randomize arrayA
    for (j = 0; j < MATRIX_SIZE; j++) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            arrayA[j * MATRIX_SIZE + i] = static_cast<SCALAR_T>(rand() % 10);
        }
    }

    // Randomize arrayB
    for (j = 0; j < MATRIX_SIZE; j++) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            arrayB[j * MATRIX_SIZE + i] = static_cast<SCALAR_T>(rand() % 10);
        }
    }

    // Initialize arrayC to 0
    for (j = 0; j < MATRIX_SIZE; j++) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            arrayC[j * MATRIX_SIZE + i] = 0;
        }
    }

    // Perform matrix multiplication on CPU
    if(ROW_MAJOR) {
        matrix_multiply_row_major_cpu(arrayA, arrayB, arrayC, MATRIX_SIZE);
    } else {
        matrix_multiply_column_major_cpu(arrayA, arrayB, arrayC, MATRIX_SIZE);
    }

    // Allocate memory on the GPU
    hipMalloc(&d_arrayA, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T));
    hipMalloc(&d_arrayB, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T));
    hipMalloc(&d_arrayC, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T));
    hipMemset(d_arrayC, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T));

    hipMemcpy(d_arrayA, arrayA, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T), hipMemcpyHostToDevice);
    hipMemcpy(d_arrayB, arrayB, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T), hipMemcpyHostToDevice);

    dim3 blockDims(THREADS_PER_2D_BLOCK, THREADS_PER_2D_BLOCK);
    dim3 gridDims(MATRIX_SIZE / THREADS_PER_2D_BLOCK + (MATRIX_SIZE % THREADS_PER_2D_BLOCK ? 1 : 0),
                  MATRIX_SIZE / THREADS_PER_2D_BLOCK + (MATRIX_SIZE % THREADS_PER_2D_BLOCK ? 1 : 0));

    // HIP kernel launch
    if(ROW_MAJOR) {
        hipLaunchKernelGGL(k_matrix_multiply_row_major, blockDims, gridDims, 0, 0, d_arrayA, d_arrayB, d_arrayC, MATRIX_SIZE);
    } else {
        hipLaunchKernelGGL(k_matrix_multiply_column_major, blockDims, gridDims, 0, 0, d_arrayA, d_arrayB, d_arrayC, MATRIX_SIZE);
    }

    hipMemcpy(arrayC_gpu, d_arrayC, MATRIX_SIZE * MATRIX_SIZE * sizeof(SCALAR_T), hipMemcpyDeviceToHost);


    cout << "Matrix Multiplication Result Comparison:\n";
    int num_diff = 0;
    for (j = 0; j < MATRIX_SIZE; j++) {
        for(i = 0; i < MATRIX_SIZE; i++)
        {
            if (arrayC[j * MATRIX_SIZE + i] != arrayC_gpu[j * MATRIX_SIZE + i]) {
                cout << "Result difference found at " << j << "," << i << ": " << "CPU: " << arrayC[j * MATRIX_SIZE + i] << " GPU: " << arrayC_gpu[j * MATRIX_SIZE + i] << ".\n";
                num_diff += 1;
            }
            else {
                // cout << "[" << j << ", " << i << "]: CPU: " << arrayC[j * MATRIX_SIZE + i] << ", GPU: " << arrayC_gpu[j * MATRIX_SIZE + i] << ".\n";
            }
        }
    }
    cout << "Number of result differences : " << num_diff << ".\n";

    hipFree(d_arrayA);
    hipFree(d_arrayB);
    hipFree(d_arrayC);

    delete[] arrayA;
    delete[] arrayB;
    delete[] arrayC;
    delete[] arrayC_gpu;

    return 0;
}