#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/utils.cuh"
#include "include/cublas_vector_scalar_add.cuh"


int main() {
    const int N = 1000000;
    const size_t size = N * sizeof(float);

    float *h_x = random_normal_clamped_array(N, -1.0f, 1.0f);
    float *d_x, *h_result;

    h_result = (float *)malloc(size);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_x, size));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    float scalar = 5.0f;

    // Call cuBLAS wrapper function
    float ms = run_vector_scalar_add(d_x, scalar, N);

    CUDA_CHECK(cudaMemcpy(h_result, d_x, size, cudaMemcpyDeviceToHost));
    verify_result(h_result, h_x, scalar, N);

    
    // Cleanup
    free(h_result);
    CUDA_CHECK(cudaFree(d_x));

    return 0;
}