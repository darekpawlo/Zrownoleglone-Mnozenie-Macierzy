#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 4  // Rozmair macierzy NxN

cudaError_t multiplyWithCuda(int* c, const int* a, const int* b, unsigned int size);

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
    int matrixSize = arraySize * arraySize;
    int* a = (int*)malloc(arraySize * arraySize * sizeof(int));
    int* b = (int*)malloc(arraySize * arraySize * sizeof(int));
    int* c = (int*)malloc(arraySize * arraySize * sizeof(int));
    cudaMalloc((void**)&d_a, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_b, matrixSize * sizeof(int));
    // Uruchomienie kernela do generowania losowych liczb
    int threadsPerBlock = 256;
    int blocksPerGrid = (matrixSize + threadsPerBlock - 1) / threadsPerBlock;
    generateRandomMatrix << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, matrixSize, time(0));
    // Synchronizacja
    cudaDeviceSynchronize();

    // Kopiowanie danych z GPU do CPU
    cudaMemcpy(a, d_a, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    // Mnożenie macierzy na GPU
    cudaError_t cudaStatus = multiplyWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyWithCuda failed!");
        return 1;
    }
    // Wyświetlenie 
    printf("Macierz A:\n");
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
    }

    // Zwalnianie pamięci
    free(a);
    free(b);
    free(c);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Funkcja pomocnicza do mnożenia macierzy z użyciem CUDA
cudaError_t multiplyWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Wybór urządzenia CUDA
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Przydzielenie pamięci na GPU dla trzech macierzy (dwie wejściowe, jedna wynikowa)
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Kopiowanie macierzy wejściowych z hosta na urządzenie (CPU -> GPU)
    cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Konfiguracja siatki i bloków wątków
    dim3 threadsPerBlock(32, 32); // Max 32x32
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    // Uruchomienie kernela na GPU
    multiplyKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);

    // Sprawdzenie, czy nie wystąpiły błędy podczas uruchamiania kernela
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Oczekiwanie na zakończenie wszystkich wątków
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyKernel!\n", cudaStatus);
        goto Error;
    }

    // Kopiowanie wynikowej macierzy z GPU na hosta
    cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
