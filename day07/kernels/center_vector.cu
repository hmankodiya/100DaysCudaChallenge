#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"


__global__ void reduce_sum(double *d_A, double *d_reduced_sum, size_t N) {
    extern __shared__ double shared_sum[];

    size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
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
        d_reduced_sum[blockIdx.x] = shared_sum[0];
}

__global__ void center_vector(double *d_A, double *d_result, double mean, size_t N) {
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < N)
        d_result[global_idx] = d_A[global_idx] - mean;

}
void verify_result(const double *h_input, const double *h_result, size_t N) {
    // Step 1: Compute mean of the input vector on CPU
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += h_input[i];
    }
    double mean = sum / N;

    // Step 2: Verify each centered value
    bool all_good = true;
    for (size_t i = 0; i < N; ++i) {
        double expected = h_input[i] - mean;
        if (std::fabs(h_result[i] - expected) > TOLERANCE) {
            std::cerr << "❌ Mismatch at index " << i
                << ": got " << h_result[i]
                << ", expected " << expected << std::endl;
            all_good = false;
            break;
        }
    }

    // Final output
    if (all_good) {
        std::cout << "✅ Centering result verification passed!" << std::endl;
    }
    else {
        std::cout << "❌ Centering result verification failed." << std::endl;
    }
}


float run_center_vector(double *d_A, double *d_result, size_t N, dim3 gridDim, dim3 blockDim) {
    double *d_reduced_sum, mean = 0.0;
    size_t _N_d_reduced_sum = gridDim.x * sizeof(d_reduced_sum);
    CUDA_CHECK(cudaMalloc(&d_reduced_sum, _N_d_reduced_sum));

    float ms = 0.0f;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // ################################################
    reduce_sum << <gridDim, blockDim >> > (d_A, d_reduced_sum, N);
    std::vector<double> h_reduced_sum(gridDim.x);

    CUDA_CHECK(cudaMemcpy(h_reduced_sum.data(), d_reduced_sum, _N_d_reduced_sum, cudaMemcpyDeviceToHost));
    cudaFree(d_reduced_sum);

    for (double val : h_reduced_sum) mean += val;
    mean /= (double)N;

    center_vector << <gridDim, blockDim >> > (d_A, d_result, mean, N);

    // ################################################
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Vector Center Kernel ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("Mean %lf \n", mean);
    printf("------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}


