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