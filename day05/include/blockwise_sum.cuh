#ifndef BLOCKWISE_SUM_CUH
#define BLOCKWISE_SUM_CUH


// total sum with strided partial sum
__global__ void block_wise_sum(float *d_A, float *d_result, int N);

int run_blockwise_sum(float *d_A, float *d_result, int N, int block_dim);

void verify_result(const float *h_partial, const float *h_input, int N, int num_blocks);

#endif