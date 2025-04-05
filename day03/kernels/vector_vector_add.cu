#include<iostream>
#include<stdio.h>
#include<string>
#include<math.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include "../include/utils.cuh"

__global__ void vector_vector_add(float *d_x, float *d_y, float *d_result, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        d_result[index] = d_x[index] + d_y[index];
}

void verify_result(const float *h_result, const float *h_x, const float *h_y, int N) {
    bool all_correct = true;
    for (int i = 0;i < N;i++) {
        float expected = h_x[i] + h_y[i];
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

float run_vector_vector_add(float *d_x, float *d_y, float *d_result, int N) {
    dim3 blockDim(256);  // reasonable block size
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);  // ceil(N / blockDim.x)

    float ms = 0.0f;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_vector_add << <gridDim, blockDim >> > (d_x, d_y, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Vector Vector Add Kernel ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}