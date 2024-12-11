#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iomanip> 
#include <cmath>
#include <vector>
#define N 10  // Size of matrices N x N
#define THREAD_BLOCK 32 // Y x Y dimensions of thread blocks
#define L 20 // Number of loops for benchmarking

// Matrix multiplication C = A x B, naive algorithm
__global__ void Naive_mat_mul(int* C, const int* A, const int* B, int n)
{
    int row = blockIdx.y * THREAD_BLOCK + threadIdx.y;
    int col = blockIdx.x * THREAD_BLOCK + threadIdx.x;

    if (row < n && col < n)
    {
        int value = 0;
        for (int k = 0; k < n; k++)
        {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

__global__ void Tiled_mat_mul(int* C, const int* A, const int* B, int n)
{
    __shared__ int s_A[THREAD_BLOCK][THREAD_BLOCK];
    __shared__ int s_B[THREAD_BLOCK][THREAD_BLOCK];

    int row = blockIdx.y * THREAD_BLOCK + threadIdx.y;
    int col = blockIdx.x * THREAD_BLOCK + threadIdx.x;
    int value = 0;

    for (int phase = 0; phase < (n + THREAD_BLOCK - 1) / THREAD_BLOCK; phase++)
    {
        int A_col = phase * THREAD_BLOCK + threadIdx.x;
        int B_row = phase * THREAD_BLOCK + threadIdx.y;

        if (row < n && A_col < n)
        {
            s_A[threadIdx.y][threadIdx.x] = A[row * n + A_col];
        }
        else
        {
            s_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < n && B_row < n)
        {
            s_B[threadIdx.y][threadIdx.x] = B[B_row * n + col];
        }
        else
        {
            s_B[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int k = 0; k < THREAD_BLOCK; k++)
        {
            value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = value;
    }
}

__global__ void generateRandomMatrix(int* a, int* b, int n, unsigned long seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * n + col;

    if (row < n && col < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        int randomValueA = (curand(&state) % 1000) + 1;
        int randomValueB = (curand(&state) % 1000) + 1;

        a[idx] = randomValueA;
        b[idx] = randomValueB;
    }
}

void rand_matrix(int* mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (int)rand() % 1000;
    }
}

void display_matrix(int* mat, int arraySize)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", mat[i * arraySize + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void check_calculations()
{
    int matrixSize = N * N * sizeof(int);

    int* h_a = (int*)malloc(matrixSize);
    int* h_b = (int*)malloc(matrixSize);
    int* h_c = (int*)malloc(matrixSize);

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, matrixSize);
    cudaMalloc(&d_b, matrixSize);
    cudaMalloc(&d_c, matrixSize);

    dim3 blockDim(THREAD_BLOCK, THREAD_BLOCK);
    dim3 gridDim((N + THREAD_BLOCK - 1) / THREAD_BLOCK, (N + THREAD_BLOCK - 1) / THREAD_BLOCK);

    generateRandomMatrix << <gridDim, blockDim >> > (d_a, d_b, N, time(NULL));
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, matrixSize, cudaMemcpyDeviceToHost);

    Tiled_mat_mul << <gridDim, blockDim >> > (d_c, d_a, d_b, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, matrixSize, cudaMemcpyDeviceToHost);

    std::cout << "Matrix A:" << std::endl;
    display_matrix(h_a, N);

    std::cout << "Matrix B:" << std::endl;
    display_matrix(h_b, N);

    std::cout << "Matrix C (Result):" << std::endl;
    display_matrix(h_c, N);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Funkcja pomocnicza do obliczenia odchylenia standardowego
double compute_std_dev(const std::vector<double>& times, double mean) {
    double sum = 0.0;
    for (auto t : times) {
        double diff = t - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum / times.size());
}

void benchmark_random_matrix_generation(int n, std::ofstream& results_file) {
    int matrixSize = n * n * sizeof(int);

    int* h_a, * h_b;
    int* d_a, * d_b;

    h_a = (int*)malloc(matrixSize);
    h_b = (int*)malloc(matrixSize);

    cudaMalloc(&d_a, matrixSize);
    cudaMalloc(&d_b, matrixSize);

    dim3 blockDim(THREAD_BLOCK, THREAD_BLOCK);
    dim3 gridDim((n + THREAD_BLOCK - 1) / THREAD_BLOCK, (n + THREAD_BLOCK - 1) / THREAD_BLOCK);

    srand((unsigned int)time(NULL));

    std::vector<double> cpu_times;
    cpu_times.reserve(L);

    // Benchmark CPU random matrix generation
    for (int i = 0; i < L; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        rand_matrix(h_a, n, n);
        rand_matrix(h_b, n, n);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        cpu_times.push_back(duration);
    }

    // Calculate CPU average and standard deviation
    double cpu_total_time = 0.0;
    for (auto t : cpu_times) cpu_total_time += t;
    double cpu_avg_time = cpu_total_time / L;
    double cpu_std_dev = compute_std_dev(cpu_times, cpu_avg_time);

    std::cout << "Average execution time (CPU) for random matrix generation for " << L << " iterations: " << cpu_avg_time << " seconds" << std::endl;
    std::cout << "Standard deviation (CPU): " << cpu_std_dev << " seconds" << std::endl;

    // Reset the CPU times vector for GPU benchmarking
    cpu_times.clear();

    std::vector<double> gpu_times;
    gpu_times.reserve(L);

    // Benchmark GPU random matrix generation
    for (int i = 0; i < L; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        generateRandomMatrix << <gridDim, blockDim >> > (d_a, d_b, n, time(NULL));
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        gpu_times.push_back(duration);
    }

    // Calculate GPU average and standard deviation
    double gpu_total_time = 0.0;
    for (auto t : gpu_times) gpu_total_time += t;
    double gpu_avg_time = gpu_total_time / L;
    double gpu_std_dev = compute_std_dev(gpu_times, gpu_avg_time);

    std::cout << "Average execution time (GPU) for random matrix generation for " << L << " iterations: " << gpu_avg_time << " seconds" << std::endl;
    std::cout << "Standard deviation (GPU): " << gpu_std_dev << " seconds" << std::endl;

    // Write results to the file with standard deviation
    results_file << n << ";" << cpu_avg_time << ";" << cpu_std_dev << ";" << gpu_avg_time << ";" << gpu_std_dev << "\n";

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
}

void benchmark_naive_matrix_mul(int n, std::ofstream& results_file, int* d_a, int* d_b, int* d_c, dim3 gridDim, dim3 blockDim) {
    for (int i = 0; i < 3; i++) {
        Naive_mat_mul << <gridDim, blockDim >> > (d_c, d_a, d_b, n);
        cudaDeviceSynchronize();
    }

    std::vector<double> iteration_times;
    iteration_times.reserve(L);

    for (int i = 0; i < L; i++)
    {
        generateRandomMatrix << <gridDim, blockDim >> > (d_a, d_b, n, time(NULL));
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        Naive_mat_mul << <gridDim, blockDim >> > (d_c, d_a, d_b, n);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        iteration_times.push_back(duration.count());
    }

    // Obliczanie średniej
    double naive_total_time = 0.0;
    for (auto t : iteration_times) naive_total_time += t;
    double naive_avg_time = naive_total_time / L;

    // Obliczanie odchylenia standardowego
    double naive_std_dev = compute_std_dev(iteration_times, naive_avg_time);

    std::cout << "Average execution time (Naive) for " << L << " iterations: " << naive_avg_time << " seconds" << std::endl;
    std::cout << "Standard deviation (Naive): " << naive_std_dev << " seconds" << std::endl;
    // Dopisujemy średnią i odchylenie do pliku
    results_file << n << ";" << naive_avg_time << ";" << naive_std_dev << ";";
}

void benchmark_tiled_matrix_mul(int n, std::ofstream& results_file, int* d_a, int* d_b, int* d_c, dim3 gridDim, dim3 blockDim) {
    for (int i = 0; i < 3; i++) {
        Tiled_mat_mul << <gridDim, blockDim >> > (d_c, d_a, d_b, n);
        cudaDeviceSynchronize();
    }

    std::vector<double> iteration_times;
    iteration_times.reserve(L);

    for (int i = 0; i < L; i++)
    {
        generateRandomMatrix << <gridDim, blockDim >> > (d_a, d_b, n, time(NULL));
        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        Tiled_mat_mul << <gridDim, blockDim >> > (d_c, d_a, d_b, n);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        iteration_times.push_back(duration.count());
    }

    // Obliczanie średniej
    double tiled_total_time = 0.0;
    for (auto t : iteration_times) tiled_total_time += t;
    double tiled_avg_time = tiled_total_time / L;

    // Obliczanie odchylenia standardowego
    double tiled_std_dev = compute_std_dev(iteration_times, tiled_avg_time);

    std::cout << "Average execution time (Tiled) for " << L << " iterations: " << tiled_avg_time << " seconds" << std::endl;
    std::cout << "Standard deviation (Tiled): " << tiled_std_dev << " seconds" << std::endl;
    // Dopisujemy średnią i odchylenie do pliku
    results_file << tiled_avg_time << ";" << tiled_std_dev << "\n";
}

void benchmark_matrix_size(int n, std::ofstream& results_file) {
    int matrixSize = n * n * sizeof(int);

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, matrixSize);
    cudaMalloc(&d_b, matrixSize);
    cudaMalloc(&d_c, matrixSize);

    dim3 blockDim(THREAD_BLOCK, THREAD_BLOCK);
    dim3 gridDim((n + THREAD_BLOCK - 1) / THREAD_BLOCK, (n + THREAD_BLOCK - 1) / THREAD_BLOCK);

    std::cout << "Benchmarking for matrix size " << n << "... " << std::endl;
    results_file << std::fixed << std::setprecision(6);

    benchmark_naive_matrix_mul(n, results_file, d_a, d_b, d_c, gridDim, blockDim);
    benchmark_tiled_matrix_mul(n, results_file, d_a, d_b, d_c, gridDim, blockDim);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    //check_calculations();

    /*std::ofstream results_file("benchmark_results.csv");
    results_file << "Matrix Size; CPU Avg Time (s); CPU Std Dev (s); GPU Avg Time (s); GPU Std Dev (s)\n";
    results_file << std::fixed << std::setprecision(6);
    benchmark_random_matrix_generation(100, results_file);
    benchmark_random_matrix_generation(500, results_file);
    benchmark_random_matrix_generation(1000, results_file);
    benchmark_random_matrix_generation(2000, results_file);
    benchmark_random_matrix_generation(5000, results_file);
    benchmark_random_matrix_generation(10000, results_file);
    benchmark_random_matrix_generation(20000, results_file);*/

    std::ofstream results_file("benchmark_results.csv");
    results_file << "Matrix Size; Naive Avg Time (s); Naive Std Dev (s); Tiled Avg Time (s); Tiled Std Dev (s)\n";
    benchmark_matrix_size(2, results_file);
    benchmark_matrix_size(10, results_file);
    benchmark_matrix_size(25, results_file);
    for (int i = 50; i < 1000; i += 50) benchmark_matrix_size(i, results_file);
    for (int i = 1000; i <= 10000; i += 100) benchmark_matrix_size(i, results_file);

    results_file.close();
    return 0;
}
