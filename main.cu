#include "CPUMatrixMultiplication.h"
#include "GPUMatrixMultiplication.h"
#include <iostream>
#include <cstdlib>
#include <chrono>

int main() {
    const int MATRIX_DIMENSION = 1024;
    const int RANGE = 100;
    const int ITERATIONS = 5;
    for (int i = 1; i <= ITERATIONS; i++) {
        int matrixSize = MATRIX_DIMENSION * i;
        int totalElements = matrixSize * matrixSize;

        // Correct memory allocation
        float* matrixA = new float[totalElements];
        float* matrixB = new float[totalElements];
        float* matrixCCPU = new float[totalElements];
        float* matrixCGPU = new float[totalElements];

        // Initialize matrices
        for (int j = 0; j < matrixSize; j++) {
            for (int k = 0; k < matrixSize; k++) {
                matrixA[j * matrixSize + k] = static_cast<float>(rand() % RANGE);
                matrixB[j * matrixSize + k] = static_cast<float>(rand() % RANGE);
            }
        }

        // CPU Matrix Multiplication Timing
        const auto startCPU = std::chrono::steady_clock::now();
        CPUMatrixMultiplication::SquareMatrixMultiplication(matrixA, matrixB, matrixCCPU, matrixSize);
        const auto endCPU = std::chrono::steady_clock::now();
        const std::chrono::duration<double> timeElapsedCPU{endCPU - startCPU};

        std::cout << "CPU Execution Time for a square matrix of size " << matrixSize
                  << ": " << timeElapsedCPU.count() << " seconds." << std::endl;

        // GPU Matrix Multiplication Timing
        const auto startGPU = std::chrono::steady_clock::now();
        GPUMatrixMultiplication::SquareMatrixMultiplication(matrixA, matrixB, matrixCGPU, matrixSize);
        const auto endGPU = std::chrono::steady_clock::now();
        const std::chrono::duration<double> timeElapsedGPU{endGPU - startGPU};

        std::cout << "GPU Execution Time for a square matrix of size " << matrixSize
                  << ": " << timeElapsedGPU.count() << " seconds.\n" << std::endl;

        // Free allocated memory

        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixCCPU;
        delete[] matrixCGPU;
    }

    return 0;
}
