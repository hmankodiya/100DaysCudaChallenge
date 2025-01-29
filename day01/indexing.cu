#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>


__global__ void printThreadIndex(int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N)
        printf("BlockIdx.x %d, ThreadIdx.x %d, index %d\n", blockIdx.x, threadIdx.x, index);
}


int main() {

    const int N = 1024;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printThreadIndex<<<blocksPerGrid, threadsPerBlock>>>(N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    return 0;
}
