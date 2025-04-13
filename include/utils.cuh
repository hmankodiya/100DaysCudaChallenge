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
float *random_normal_clamped_array(size_t N, float min_val, float max_val);

// ===== FLOAT =====
float *ones_arr_float(size_t N);
float *arange_float(size_t low, size_t high, size_t N);
float *arange_float(size_t high, size_t N);
void display_float(float *arr, size_t N);

// ===== DOUBLE =====
double *ones_arr_double(size_t N);
double *arange_double(double low, double high, size_t N);
double *arange_double(double high, size_t N);
void display_double(double *arr, size_t N);

// ===== SIZE_T =====
size_t *ones_arr_size_t(size_t N);
size_t *arange_size_t(size_t low, size_t high, size_t N);
size_t *arange_size_t(size_t high, size_t N);
void display_size_t(size_t *arr, size_t N);


#endif
