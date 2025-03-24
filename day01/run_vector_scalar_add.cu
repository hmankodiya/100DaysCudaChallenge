// main.cu

#include <iostream>
#include "../include/utils.cuh"
#include "./include/vector_scalar_add.cuh"

int main() {
    // Matrix dimensions
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Allocate and initialize host matrices
    float val = 100;
    float *h_A = random_normal_clamped_array(N, -1.0f, 1.0f);
    float *h_result;
    h_result = (float *)malloc(size);

    // Allocate device memory
    float *d_A, *d_result;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_result, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Run the kernel
    float ms = run_vector_scalar_add(d_A, d_result, val, N);

    CUDA_CHECK(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));
    verify_result(h_result, h_A, val, N);

    // Cleanup
    free(h_A);
    free(h_result);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
