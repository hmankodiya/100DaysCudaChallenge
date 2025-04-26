#ifndef BLOCKWISE_SUM_CUH
#define BLOCKWISE_SUM_CUH


// total sum with strided partial sum
__global__ void center_vector(double *d_A, double *d_result, double mean, size_t N);

float run_center_vector(double *d_A, double *d_result, size_t N, dim3 gridDim, dim3 blockDim);
void verify_result(const double *h_input, const double *h_result, size_t N);

#endif