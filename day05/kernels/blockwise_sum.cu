#include <iostream>
#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"


// total sum with strided partial sum
__global__ void block_wise_sum(float *d_A, float *d_result, int N) {
    extern __shared__ float shared_sum[];

    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    int local_index = threadIdx.x;

    shared_sum[local_index] = (global_index < N) ? d_A[global_index] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_sum[local_index] += shared_sum[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        printf("[blockIdx.x %d, threadIdx.x %d, blockDim.x %d] => global_index %d, local_index %d, d_A[global_index] = %f, shared_sum[local_index] = %f\n", blockIdx.x, threadIdx.x, blockDim.x, global_index, local_index, d_A[global_index], shared_sum[local_index]);
        d_result[blockIdx.x] = shared_sum[0];
    }

}

void verify_result(const float *h_partial, const float *h_input, int N, int num_blocks) {
    // Compute final result from GPU partials
    float gpu_total = 0.0f;
    for (int i = 0; i < num_blocks; ++i) {
        gpu_total += h_partial[i];
    }

    // Compute reference sum on CPU
    float expected_total = 0.0f;
    for (int i = 0; i < N; ++i) {
        expected_total += h_input[i];
    }

    // Compare
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

int run_blockwise_sum(float *d_A, float *d_result, int N, int block_dim) {
    dim3 blockDim(block_dim);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    float ms = 0.0f;
    cudaEvent_t start, stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    block_wise_sum << <gridDim, blockDim >> > (d_A, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Vector Vector Add Kernel ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // blockwise sum results
    return 0;
}


