#include "CPUMatrixMultiplication.h"
#include <vector>

auto CPUMatrixMultiplication::SquareMatrixMultiplication(const float* matrixA, const float* matrixB, float* matrixC, int matrixDimension) -> void {
    for (int i = 0; i < matrixDimension; i++)
    {
        for (int j = 0; j < matrixDimension; j++)
        {
            float value = 0;
            for (int k = 0; k < matrixDimension; k++)
            {
                value += matrixA[i * matrixDimension + k] * matrixB[k * matrixDimension + j];
            }

            matrixC[i * matrixDimension + j] = value;
        }
    }
}

CPUMatrixMultiplication::CPUMatrixMultiplication() {}