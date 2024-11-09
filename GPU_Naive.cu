//naive
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#define N 20000  // Rozmair macierzy NxN
#define TILE_WIDTH 16

__global__ void multiplyKernel(int* C, const int* A, const int* B, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int value = 0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}
// Kernel do generowania losowych liczb dla macierzy a i b
__global__ void generateRandomMatrix(int* a, int* b, int size, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generowanie liczb pseudolosowych w przedziale 1-10
        int randomValueA = (curand(&state) % 10) + 1;
        int randomValueB = (curand(&state) % 10) + 1;

        a[idx] = randomValueA;
        b[idx] = randomValueB;
    }
}
int main()
{
    const int arraySize = N;
    int* d_a;
    int* d_b;
    int* d_c;
    int matrixSize = arraySize * arraySize;
    int* a = (int*)malloc(arraySize * arraySize * sizeof(int));
    int* b = (int*)malloc(arraySize * arraySize * sizeof(int));
    int* c = (int*)malloc(arraySize * arraySize * sizeof(int));
    cudaMalloc((void**)&d_a, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_b, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_c, matrixSize * sizeof(int));
    // Uruchomienie kernela do generowania losowych liczb
    int threadsPerBlock = 256;
    int blocksPerGrid = (matrixSize + threadsPerBlock - 1) / threadsPerBlock;
    generateRandomMatrix << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, matrixSize, time(0));
    // Synchronizacja
    cudaDeviceSynchronize();

    // Kopiowanie danych z GPU do CPU
    cudaMemcpy(a, d_a, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    auto start = std::chrono::high_resolution_clock::now();
    // Mnożenie macierzy na GPU
   // Konfiguracja siatki i bloków wątków
    dim3 threadsPerBlockk(TILE_WIDTH, TILE_WIDTH); // Max 32x32
    dim3 blocksPerGridd((N + threadsPerBlockk.x - 1) / threadsPerBlockk.x, (N + threadsPerBlockk.y - 1) / threadsPerBlockk.y);


    // Uruchomienie kernela na GPU
    multiplyKernel << <blocksPerGridd, threadsPerBlockk >> > (d_c, d_a, d_b, arraySize);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cudaMemcpy(c, d_c, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Czas wykonania tailed: " << duration.count() << " milliseconds" << std::endl;
    // Wyświetlenie 
   /* printf("Macierz A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", a[i * arraySize + j]);
        }
        printf("\n");
    }
    printf("Macierz B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", b[i * arraySize + j]);
        }
        printf("\n");
    }
    printf("Macierz C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i * arraySize + j]);
        }
        printf("\n");
    }*/

    // Zwalnianie pamięci
    free(a);
    free(b);
    free(c);

    return 0;
}
