#ifndef VECTOR_VECTOR_ADD_CUH
#define VECTOR_VECTOR_ADD_CUH

__global__ void vector_vector_add(float *d_x, float *d_y, float *d_result, int N);

void verify_result(const float *h_result, const float *h_x, const float *h_y, int N);

float run_vector_vector_add(float *d_x, float *d_y, float *d_result, int N);

#endif