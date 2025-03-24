#ifndef VECTOR_SCALAR_ADD_CUH
#define VECTOR_SCALAR_ADD_CUH

__global__ void vector_scalar_add(float *d_vector, float *d_result, float val, int N);

float run_vector_scalar_add(float *d_vector, float *d_result, float val, int N);

void verify_result(const float *h_result, const float *h_vector, float val, int N);

#endif