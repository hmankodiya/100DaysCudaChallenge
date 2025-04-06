
#include <iostream>
#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"


// thread_safe block_sum, and dynamic shared memory
__global__ void block_wise_sum(float *d_A, int N) {
    extern __shared__ float shared_sum[];

    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    int local_index = threadIdx.x;

    shared_sum[local_index] = (global_index < N) ? d_A[global_index] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_sum[local_index] += shared_sum[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0)
        printf("[blockIdx.x %d, threadIdx.x %d, blockDim.x %d] => global_index %d, local_index %d, d_A[global_index] = %f, shared_sum[local_index] = %f\n", blockIdx.x, threadIdx.x, blockDim.x, global_index, local_index, d_A[global_index], shared_sum[local_index]);

}
