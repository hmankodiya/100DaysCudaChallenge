#include <iostream>
#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"

// this wont work
__global__ void _nonthread_safe_block_wise_sum(float *d_A, int N) { // the shared memory is shared in a block but it is not thread-safe
    __shared__ float shared_sum;
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadIdx.x == 0)
        shared_sum = 0.0f;

    __syncthreads();

    if (index < N)
        shared_sum += d_A[index];

    __syncthreads();

    if (index < N) {
        printf("[blockIdx.x %d, threadIdx.x %d, blockDim.x %d] => index %d, d_A[index] = %f, shared_sum = %f\n", blockIdx.x, threadIdx.x, blockDim.x, index, d_A[index], shared_sum);
    }

}
