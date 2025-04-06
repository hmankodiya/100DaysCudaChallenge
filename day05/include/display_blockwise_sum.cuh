#ifndef DISPLAY_BLOCKWISE_SUM_CUH
#define DISPLAY_BLOCKWISE_SUM_CUH 


// thread_safe block_sum, and dynamic shared memory
__global__ void block_wise_sum(float *d_A, int N);

#endif