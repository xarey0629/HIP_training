#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <vector>
using namespace std;

#define MATRIX_SIZE 1024
#define SCALAR_T float
#define THREADS_PER_2D_BLOCK 32

template <typename T>
__global__ void k_matrix_multiply(T* a, T* b, T* out, size_t dim) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < dim  && y < dim) {
        T out_k = 0;
        for (int k = 0; k < dim; k++)
            out_k += a[y * dim + k] * b[k * dim + x];
        out[y * dim + x] = out_k;
    }
}

int main()
{
    // vector<SCALAR_T> arrayA(MATRIX_SIZE * MATRIX_SIZE);     // Matrix A
    // vector<SCALAR_T> arrayB(MATRIX_SIZE * MATRIX_SIZE);     // Matrix B
    // vector<SCALAR_T> arrayC(MATRIX_SIZE * MATRIX_SIZE, 0);     // Matrix C (for CPU)
    // vector<SCALAR_T> arrayC_gpu(MATRIX_SIZE * MATRIX_SIZE); // Matrix C (for GPU returns)
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

    // Matrix Multiplication on CPU
    for (j = 0; j < MATRIX_SIZE; j++) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (k = 0; k < MATRIX_SIZE; k++) {
                arrayC[j * MATRIX_SIZE + i] += arrayA[j * MATRIX_SIZE + k] * arrayB[k * MATRIX_SIZE + i];
            }
        }
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
    hipLaunchKernelGGL(k_matrix_multiply, blockDims, gridDims, 0, 0, d_arrayA, d_arrayB, d_arrayC, MATRIX_SIZE);

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
                cout << "[" << j << ", " << i << "]: CPU: " << arrayC[j * MATRIX_SIZE + i] << ", GPU: " << arrayC_gpu[j * MATRIX_SIZE + i] << ".\n";
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