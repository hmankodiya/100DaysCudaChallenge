// include/utils.cuh

#ifndef UTILS_CUH
#define UTILS_CUH

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>

#define TOLERANCE 1e-4

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

// Device + Host random number function
float random_normal_clamped(float min_val, float max_val);
float *random_normal_clamped_array(int N, float min_val, float max_val);

#endif // UTILS_CUH
