#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/utils.cuh"

void verify_result(const float *h_result, const float *h_x, const float scalar, int N) {
    bool all_correct = true;
    for (int i = 0;i < N;i++) {
        float expected = h_x[i] + scalar;
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


float *cublas_vector_scalar_add(cublasHandle_t handle, float *d_x, float scalar, int N) {
    float *h_ones = ones_arr(N);  // host ones array
    size_t size = N * sizeof(float);

    float *d_ones;
    CUDA_CHECK(cudaMalloc((void **)&d_ones, size));
    CUDA_CHECK(cudaMemcpy(d_ones, h_ones, size, cudaMemcpyHostToDevice));

    // Perform: d_x = d_x + scalar * d_ones
    CUBLAS_CHECK(cublasSaxpy(handle, N, &scalar, d_ones, 1, d_x, 1));

    // Allocate space for result on host and copy back
    float *h_result = (float *)malloc(size);
    CUDA_CHECK(cudaMemcpy(h_result, d_x, size, cudaMemcpyDeviceToHost));

    // Cleanup
    free(h_ones);
    CUDA_CHECK(cudaFree(d_ones));

    return h_result;  // Caller is responsible for freeing h_result
}


float run_vector_scalar_add(float *d_x, float scalar, int N) {
    // dim3 blockDim(256);
    // dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    float ms = 0.0f;
    cudaEvent_t start, stop;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    cublas_vector_scalar_add(handle, d_x, scalar, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("------- Vector Scalar Add Kernel ---------\n");
    printf("Elapsed time: %f ms\n", ms);
    printf("------------------------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    return ms;
}