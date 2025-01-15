#pragma once

class GPUMatrixMultiplication {
public:
  static const int BLOCK_SIZE = 8;
  static auto SquareMatrixMultiplication(const float* matrixA, const float* matrixB, float* matrixC, int matrixDimension) -> void;
  
private:
  GPUMatrixMultiplication();
};