#ifndef BLOCKWISE_SUM_CUH
#define BLOCKWISE_SUM_CUH


// total sum with strided partial sum
__global__ void partial_sum(double *d_A, double *d_result, size_t N);
__global__ void partial_sum_int(size_t *d_A, size_t *d_result, size_t N);

size_t run_partial_sum(double *d_A, double *d_result, size_t N, size_t block_dim);
size_t run_partial_sum_int(size_t *d_A, size_t *d_result, size_t N, size_t block_dim);

void verify_result(const double *h_input, const double *h_result, size_t N, size_t num_blocks);
void verify_result_int(const size_t *h_input, const size_t *h_result, size_t N, size_t num_blocks);

#endif