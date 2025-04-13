#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"
#include "./include/partial_sum.cuh"

int main() {
    // Input parameters
    size_t N;
    printf("array size: ");
    scanf("%ld", &N);

    size_t block_dim;
    printf("block_dim: ");
    scanf("%ld", &block_dim);

    const size_t size = N * sizeof(size_t);

    // Allocate and initialize host input
    size_t *h_A = (size_t *)malloc(size);
    for (size_t i = 0; i < N; ++i)
        h_A[i] = i + 1;

    if (N < 20)
        display_size_t(h_A, N);  // Overloaded display for int in utils.cuh

    // Grid/block setup
    dim3 blockDim(block_dim);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    size_t size_result = gridDim.x * sizeof(size_t);

    // Allocate device memory
    size_t *d_A, *d_result, *h_result;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    h_result = (size_t *)malloc(size_result);
    CUDA_CHECK(cudaMalloc((void **)&d_result, size_result));

    // Run the kernel
    size_t shared_mem_bytes = block_dim * sizeof(size_t);
    partial_sum_int<<<gridDim, blockDim, shared_mem_bytes>>>(d_A, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost));

    // Display
    if (N < 20)
        display_size_t(h_result, gridDim.x);

    // Verify
    size_t gpu_total = 0, cpu_total = 0;
    for (size_t i = 0; i < gridDim.x; ++i)
        gpu_total += h_result[i];
    for (size_t i = 0; i < N; ++i)
        cpu_total += h_A[i];

    if (gpu_total != cpu_total) {
        std::cerr << "❌ Total sum mismatch: got " << gpu_total
                  << ", expected " << cpu_total << std::endl;
    } else {
        std::cout << "✅ Result verification passed!" << std::endl;
        std::cout << "✅ GPU Total: " << gpu_total << ", CPU Total: " << cpu_total << std::endl;
    }

    // Cleanup
    free(h_A);
    free(h_result);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
