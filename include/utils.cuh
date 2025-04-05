// include/utils.cuh

#ifndef UTILS_CUH
#define UTILS_CUH

#include<stdio.h>
#include<cuda.h>
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>

#define TOLERANCE 1e-5

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
            cudaGetErrorName(code), cudaGetErrorString(code),
            file, line);
        exit(code);
    }
}

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Device + Host random number function
float random_normal_clamped(float min_val, float max_val);
float *random_normal_clamped_array(int N, float min_val, float max_val);

float *ones_arr(int N);

#endif // UTILS_CUH
