#include<iostream>
#include<cuda_runtime.h>
#include<time.h>
#include<stdlib.h>

using namespace std;


#define N 10000000
#define BLOCK_SIZE_1D 1024


__device__ float square(float value) {
    return value * value;
}

__global__ void square_kernel(float *x, float *y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = square(x[index]);
    }
}

void init_vector(float *vec, int n) {
    for (int i = 0;i < n;i++)
        vec[i] = (float)rand() / RAND_MAX;
}


int main() {

    float *h_a, *h_b;
    float *d_a, *d_b;

    size_t size = N * sizeof(float);

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);


    srand(time(NULL));
    init_vector(h_a, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int blocksPerGrid = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    square_kernel << <blocksPerGrid, BLOCK_SIZE_1D >> > (d_a, d_b, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);


    free(h_a);
    cudaFree(d_a);


    return 0;
}