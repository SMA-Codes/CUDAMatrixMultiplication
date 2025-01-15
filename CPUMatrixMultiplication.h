#pragma once

class CPUMatrixMultiplication {
public:
  static auto SquareMatrixMultiplication(const float* matrixA, const float* matrixB, float* matrixC, int matrixDimension) -> void;
  
private:
  CPUMatrixMultiplication();
};