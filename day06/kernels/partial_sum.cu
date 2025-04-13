#include <iostream>
#include <stdio.h>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"


__global__ void partial_sum(double *d_A, double *d_result, size_t N) {
    extern __shared__ double shared_sum[];
    // __shared__ double shared_sum[128];

    size_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t local_index = threadIdx.x;

    shared_sum[local_index] = (global_index < N) ? d_A[global_index] : (double)0.0;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_sum[local_index] += shared_sum[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0)
        d_result[blockIdx.x] = shared_sum[0];

}

void verify_result_int(const size_t *h_input, const size_t *h_result, size_t N, size_t num_blocks) {
    size_t gpu_total = 0;
    for (size_t i = 0; i < num_blocks; ++i)
        gpu_total += h_result[i];

    size_t expected_total = 0;
    for (size_t i = 0; i < N; ++i)
        expected_total += h_input[i];

    if (gpu_total != expected_total) {
        std::cerr << "❌ Total sum mismatch: got " << gpu_total
            << ", expected " << expected_total << std::endl;
        std::cout << "❌ Result verification failed." << std::endl;
    }
    else {
        std::cout << "✅ Result verification passed!" << std::endl;
        std::cout << "✅ GPU Total: " << gpu_total << ", CPU Total: " << expected_total << std::endl;
    }
}


void verify_result(const double *h_input, const double *h_result, size_t N, size_t num_blocks) {
    // Compute final result from GPU partials
    double gpu_total = (double)0.0;
    for (size_t i = 0; i < num_blocks; ++i) {
        gpu_total += h_result[i];
    }

    // Compute reference sum on CPU
    double expected_total = (double)0.0;
    for (size_t i = 0; i < N; ++i) {
        expected_total += h_input[i];
    }

    if (std::fabs(gpu_total - expected_total) > TOLERANCE) {
        std::cerr << "❌ Total sum mismatch: got " << gpu_total
            << ", expected " << expected_total << std::endl;
        std::cout << "❌ Result verification failed." << std::endl;
    }
    else {
        std::cout << "✅ Result verification passed!" << std::endl;
        std::cout << "✅ GPU Total: " << gpu_total << ", CPU Total: " << expected_total << std::endl;
    }
}

__global__ void partial_sum_int(size_t *d_A, size_t *d_result, size_t N) {
    extern __shared__ size_t shared_sum_int[];

    size_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t local_index = threadIdx.x;

    // Load from global memory to shared memory
    shared_sum_int[local_index] = (global_index < N) ? d_A[global_index] : 0;
    __syncthreads();

    // Reverse-stride tree reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_sum_int[local_index] += shared_sum_int[local_index + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (local_index == 0)
        d_result[blockIdx.x] = shared_sum_int[0];
}


int run_partial_sum_int(size_t *d_A, size_t *d_result, size_t N, size_t block_dim) {
    dim3 blockDim(block_dim);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    int shared_mem_size = blockDim.x * sizeof(size_t);

    float ms = 0.0f;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    partial_sum_int << <gridDim, blockDim, shared_mem_size >> > (d_A, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Partial-Sum Add Kernel (int) ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("----------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}


float run_partial_sum(double *d_A, double *d_result, size_t N, size_t block_dim) {
    dim3 blockDim(block_dim);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    size_t shared_mem_size = blockDim.x * sizeof(double);


    float ms = 0.0f;
    cudaEvent_t start, stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    partial_sum << <gridDim, blockDim, shared_mem_size >> > (d_A, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Partial-Sum Add Kernel ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}


