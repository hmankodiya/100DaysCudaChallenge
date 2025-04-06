#include <iostream>
#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"
#include "./include/blockwise_sum.cuh"


int main() {
    // Matrix dimensions
    int N;
    printf("array size:");
    scanf("%d", &N);

    int block_dim;
    printf("block_dim:");
    scanf("%d", &block_dim);

    const size_t size = N * sizeof(float);

    // Allocate and initialize host matrices
    float *h_A = arange(0, N, N);
    if (N < 20)
        display(h_A, N);

    dim3 blockDim(block_dim);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Allocate device memory
    float *d_A, *d_result, *h_result;
    size_t size_result = gridDim.x * sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)); // Copy data from host to device

    h_result = (float *)malloc(size_result);
    CUDA_CHECK(cudaMalloc((void **)&d_result, size_result));

    // Run the kernel
    float ms = run_blockwise_sum(d_A, d_result, N, block_dim);

    CUDA_CHECK(cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost)); // Copy data from host to device

    // blockwise sum results
    if (N < 20)
        display(h_result, gridDim.x);

    verify_result(h_result, h_A, N, gridDim.x);

    // Cleanup
    free(h_A);
    free(h_result);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
