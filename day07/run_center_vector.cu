#include <iostream>
#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"
#include "./include/center_vector.cuh"


int main() {
    // Matrix dimensions
    int N;
    printf("array size:");
    scanf("%d", &N);

    int block_dim;
    printf("block_dim:");
    scanf("%d", &block_dim);

    const size_t size = N * sizeof(double);

    // Allocate and initialize host matrices
    double *h_A = arange_double(0, N, N);
    // double *h_A = ones_arr_double(N);
    if (N < 20)
        display_double(h_A, N);

    dim3 blockDim(block_dim);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Allocate device memory
    double *d_A, *d_result, *h_result;

    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)); // Copy data from host to device

    h_result = (double *)malloc(size);
    CUDA_CHECK(cudaMalloc((void **)&d_result, size));

    // Run the kernel
    float ms = run_center_vector(d_A, d_result, N, gridDim, blockDim);

    CUDA_CHECK(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    // blockwise sum results
    if (N < 20)
        display_double(h_result, N);

    verify_result(h_A, h_result, N);

    // Cleanup
    free(h_A);
    free(h_result);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
