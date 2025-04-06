// utils.cu

#include <cstdlib>
#include <cmath>
#include "utils.cuh"


// CPU + GPU implementation of clamped random float
float random_normal_clamped(float min_val, float max_val) {
    float r = ((float)rand() / RAND_MAX);  // Uniform in [0, 1]
    return min_val + r * (max_val - min_val);
}

float *random_normal_clamped_array(int N, float min_val, float max_val) {
    float *arr;
    arr = (float *)malloc(N * sizeof(N));
    for (size_t i = 0;i < N;i++)
        arr[i] = random_normal_clamped(min_val, max_val);

    return arr;
}

float *ones_arr(int N) {
    float *arr;
    arr = (float *)malloc(N * sizeof(N));
    for (size_t i = 0;i < N;i++)
        arr[i] = 1.0f;

    return arr;
}

float *arange(int low, int high, int N) {
    float *arr = (float *)malloc(N * sizeof(N));
    for (size_t i = 0;i < N;i++)
        arr[i] = low + i * (high - low) / (float)N;

    return arr;
}

float *arange(int high, int N) {
    return arange(0, high, N);
}


void display(float *arr, int N) {
    for (size_t i = 0;i < N;i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}