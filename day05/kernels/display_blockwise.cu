#include <iostream>
#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/utils.cuh"


__global__ void display_block_wise(float *d_A, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if (index < N)
        printf("[blockIdx.x %d, threadIdx.x %d, blockDim.x %d] => index %d, value = %f\n",
            blockIdx.x, threadIdx.x, blockDim.x, index, d_A[index]);
}
