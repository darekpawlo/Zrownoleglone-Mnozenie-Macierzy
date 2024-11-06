#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 2000  // Rozmiar macierzy NxN
#define TILE_WIDTH 16

cudaError_t multiplyWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void tiledMultiplyKernel(int* C, const int* A, const int* B, int n)
{
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;   int by = blockIdx.y;
    int tx = threadIdx.x;  int ty = threadIdx.y;

    // Identyfikacja wiersza i kolumny elementu C do obliczenia
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int Pvalue = 0;

    // Pętla po wszystkich kafelkach wymaganych do obliczenia elementu C
    for (int m = 0; m < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        // Współdzielone ładowanie kafelków do pamięci współdzielonej
        if (Row < n && m * TILE_WIDTH + tx < n)
            ds_A[ty][tx] = A[Row * n + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0;

        if (Col < n && m * TILE_WIDTH + ty < n)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * n + Col];
        else
            ds_B[ty][tx] = 0;

        __syncthreads();

        // Mnożenie dwóch kafelków
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < n && Col < n)
        C[Row * n + Col] = Pvalue;
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
    int* a = (int*)malloc(matrixSize * sizeof(int));
    int* b = (int*)malloc(matrixSize * sizeof(int));
    int* c = (int*)malloc(matrixSize * sizeof(int));
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

    // Wyświetlenie części macierzy (dla dużych N ograniczamy wyświetlanie)
    int displaySize = (N > 10) ? 10 : N;
    printf("Macierz A:\n");
    for (int i = 0; i < displaySize; i++) {
        for (int j = 0; j < displaySize; j++) {
            printf("%d ", a[i * arraySize + j]);
        }
        printf("\n");
    }
    if (N > 10) printf("...\n");

    printf("Macierz B:\n");
    for (int i = 0; i < displaySize; i++) {
        for (int j = 0; j < displaySize; j++) {
            printf("%d ", b[i * arraySize + j]);
        }
        printf("\n");
    }
    if (N > 10) printf("...\n");

    printf("Macierz C:\n");
    for (int i = 0; i < displaySize; i++) {
        for (int j = 0; j < displaySize; j++) {
            printf("%d ", c[i * arraySize + j]);
        }
        printf("\n");
    }
    if (N > 10) printf("...\n");

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
        fprintf(stderr, "cudaSetDevice failed! Czy masz zainstalowaną kartę CUDA?");
        goto Error;
    }

    // Przydzielenie pamięci na GPU dla trzech macierzy
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
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);

    // Uruchomienie kernela na GPU
    tiledMultiplyKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);

    // Sprawdzenie błędów podczas uruchamiania kernela
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "tiledMultiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Oczekiwanie na zakończenie wszystkich wątków
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching tiledMultiplyKernel!\n", cudaStatus);
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
