#include "GPUMatrixMultiplication.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16 // Ensure this is a supported size for your GPU

// CUDA Kernel
__global__ void MatrixMultiplicationKernel(const float* matrixA, const float* matrixB, float* matrixC, int matrixDimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrixDimension && col < matrixDimension) {
        float sum = 0.0f;
        for (int i = 0; i < matrixDimension; i++) {
            sum += matrixA[row * matrixDimension + i] * matrixB[i * matrixDimension + col];
        }
        matrixC[row * matrixDimension + col] = sum;
    }
}


// Host Function
auto GPUMatrixMultiplication::SquareMatrixMultiplication(const float* matrixA, const float* matrixB, float* matrixC, int matrixDimension) -> void {
    float *d_matrixA, *d_matrixB, *d_matrixC;

    size_t matrixSize = matrixDimension * matrixDimension * sizeof(float);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (3 * matrixSize > freeMem) {
        std::cerr << "Insufficient GPU memory for this matrix size!" << std::endl;
        return;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_matrixA, matrixSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_matrixA: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc((void**)&d_matrixB, matrixSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_matrixB: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrixA);
        return;
    }

    err = cudaMalloc((void**)&d_matrixC, matrixSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_matrixC: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        return;
    }

    // Copy host data to device
    err = cudaMemcpy(d_matrixA, matrixA, matrixSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_matrixA: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);
        return;
    }

    err = cudaMemcpy(d_matrixB, matrixB, matrixSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_matrixB: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);
        return;
    }

    // Define thread block and grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((matrixDimension + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (matrixDimension + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 1);

    // Launch kernel
    MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(d_matrixA, d_matrixB, d_matrixC, matrixDimension);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);
        return;
    }

    // Copy result back to host
    err = cudaMemcpy(matrixC, d_matrixC, matrixSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_matrixC: " << cudaGetErrorString(err) << std::endl;
    }

    // Free device memory
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
}
