#ifndef NONTHREAD_SAFE_BLOCKWISE_SUM_CUH
#define NONTHREAD_SAFE_BLOCKWISE_SUM_CUH

// this wont work
__global__ void _nonthread_safe_block_wise_sum(float *d_A, int N);

#endif