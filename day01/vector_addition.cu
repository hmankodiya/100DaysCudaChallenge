#include<iostream>
#include<cuda_runtime.h>
#include<time.h>
#include<stdlib.h>

using namespace std;


#define N 10000000
#define BLOCK_SIZE_1D 1024


__global__ void add(float *a, float *b, float *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

void init_vector(float *vec, int n) {
    for (int i = 0;i < n;i++)
        vec[i] = (float)rand() / RAND_MAX;
}

int main() {

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blocksPerGrid = (N + BLOCK_SIZE_1D - 1) / (BLOCK_SIZE_1D);
    add << <blocksPerGrid, BLOCK_SIZE_1D >> > (d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}