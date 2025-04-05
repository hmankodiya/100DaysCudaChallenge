#ifndef CUBLAS_VEC_SCALAR_ADD_CUH
#define CUBLAS_VEC_SCALAR_ADD_CUH

#include<cublas_v2.h>

float *cublas_vector_scalar_add(cublasHandle_t handle, float *d_x, float scalar, int N);
void verify_result(const float *h_result, const float *h_x, const float scalar, int N);
float run_vector_scalar_add(float *d_x, float scalar, int N);

#endif