#include<iostream>
#include<stdio.h>
#include<string>
#include<math.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include "../include/utils.cuh"

__global__ void matrix_scalar_add(float *d_matrix, float *d_result, float val, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int index = row * M + col;
        d_result[index] = d_matrix[index] + val;
    }
}

void verify_result(const float *h_result, const float *h_matrix, float val, int N, int M) {
    bool all_correct = true;
    for (int i = 0; i < N * M; ++i) {
        float expected = h_matrix[i] + val;
        if (fabs(h_result[i] - expected) > TOLERANCE) {
            std::cerr << "❌ Mismatch at index " << i
                << ": got " << h_result[i]
                << ", expected " << expected << std::endl;
                all_correct = false;
                break;
        }
    }

    if (all_correct) {
        std::cout << "✅ Result verification passed!" << std::endl;
    }
    else {
        std::cout << "❌ Result verification failed." << std::endl;
    }
}


float run_matrix_scalar_add(float *d_matrix, float *d_result, float val, int N, int M) {
    dim3 blockDim(16, 16);  // reasonable block size
    dim3 gridDim(((N + blockDim.x - 1) / blockDim.x), ((M + blockDim.y - 1) / blockDim.y));  // ceil(N / blockDim.x)

    float ms = 0.0f;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrix_scalar_add << <gridDim, blockDim >> > (d_matrix, d_result, val, N, M);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Matrix Scalar Add Kernel ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}
