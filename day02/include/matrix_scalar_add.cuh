#ifndef MATRIX_SCALAR_ADD_CUH
#define MATRIX_SCALAR_ADD_CUH

__global__ void matrix_scalar_add(float *d_matrix, float *d_result, float val, int N, int M);

float run_matrix_scalar_add(float *d_matrix, float *d_result, float val, int N, int M);

void verify_result(const float *h_result, const float *h_matrix, float val, int N, int M);

#endif