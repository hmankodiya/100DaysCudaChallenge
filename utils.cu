// utils.cu

#include <cstdlib>
#include <cmath>
#include "utils.cuh"


// CPU + GPU implementation of clamped random float
float random_normal_clamped(float min_val, float max_val) {
    float r = ((float)rand() / RAND_MAX);  // Uniform in [0, 1]
    return min_val + r * (max_val - min_val);
}

float *random_normal_clamped_array(size_t N, float min_val, float max_val) {
    float *arr;
    arr = (float *)malloc(N * sizeof(N));
    for (size_t i = 0;i < N;i++)
        arr[i] = random_normal_clamped(min_val, max_val);

    return arr;
}



// ===== FLOAT =====
float *ones_arr_float(size_t N) {
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = 1.0f;
    return arr;
}

float *arange_float(size_t low, size_t high, size_t N) {
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = low + i * (high - low) / (float)N;
    return arr;
}

float *arange_float(size_t high, size_t N) {
    return arange_float(0, high, N);
}

void display_float(float *arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}


// ===== DOUBLE =====
double *ones_arr_double(size_t N) {
    double *arr = (double *)malloc(N * sizeof(double));
    for (size_t i = 0; i < N; i++)
        arr[i] = 1.0;
    return arr;
}

double *arange_double(double low, double high, size_t N) {
    double *arr = (double *)malloc(N * sizeof(double));
    for (size_t i = 0; i < N; i++)
        arr[i] = low + i * (high - low) / (double)N;
    return arr;
}

double *arange_double(double high, size_t N) {
    return arange_double(0.0, high, N);
}

void display_double(double *arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}


// ===== SIZE_T =====
size_t *ones_arr_size_t(size_t N) {
    size_t *arr = (size_t *)malloc(N * sizeof(size_t));
    for (size_t i = 0; i < N; i++)
        arr[i] = 1;
    return arr;
}

size_t *arange_size_t(size_t low, size_t high, size_t N) {
    size_t *arr = (size_t *)malloc(N * sizeof(size_t));
    for (size_t i = 0; i < N; i++)
        arr[i] = low + i * (high - low) / N;
    return arr;
}

size_t *arange_size_t(size_t high, size_t N) {
    return arange_size_t(0, high, N);
}

void display_size_t(size_t *arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        printf("%lu ", arr[i]);
    }
    printf("\n");
}