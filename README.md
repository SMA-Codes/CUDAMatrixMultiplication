# CUDA Matrix Multiplication

## Overview

This project demonstrates GPU-accelerated matrix multiplication using CUDA and C++. By leveraging CUDA’s parallel computing capabilities, this implementation achieves significant performance improvements over CPU-based matrix multiplication.

## Features

- Supports matrices of size \(N 	imes M\), with dimensions optimized for GPU thread blocks (multiples of 16).
- Highly parallelized computation using CUDA kernels.
- Performance comparison between GPU and CPU implementations.
- Configurable matrix sizes for experimentation and benchmarking.

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher.
- NVIDIA CUDA Toolkit.
- C++ compiler supporting C++11 or later.

## Installation

### Prerequisites

1. **CUDA Toolkit**:
   - Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) for your operating system.

2. **C++ Compiler**:
   - Ensure your system has a compatible compiler, such as `gcc` or `clang` (Linux/macOS) or MSVC (Windows).

### Build Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/SMA-Codes/CUDAMatrixMultiplication.git
   cd CUDAMatrixMultiplication
   ```

2. Compile the program:
   ```bash
   nvcc -o cuda_matrix_mul matrix_mul.cu
   ```

3. Run the program:
   ```bash
   ./cuda_matrix_mul
   ```

## Usage

1. Modify the matrix dimensions in `matrix_mul.cu` if needed (ensure dimensions are multiples of 16):
   ```cpp
   const int N = 512;  // Number of rows
   const int M = 512;  // Number of columns
   ```

2. Compile and run the program to see the GPU and CPU performance comparison.

3. Output includes:
   - Execution time for both GPU and CPU matrix multiplication.
   - Speedup achieved by using CUDA.


## Performance

The CUDA implementation significantly outperforms the CPU implementation, especially for large matrices. For example:

| Matrix Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|-------------|---------------|---------------|---------|
| 1024 x 1024 | 500           | 0.25          | 2000x   |
| 2048 x 2048 | 4000          | 1.5           | 2667x   |

## Future Enhancements

- Add support for non-square matrices.
- Implement shared memory optimization for further speed improvements.
- Extend benchmarking for various GPU architectures.
- Add visualization of computation using tools like Nsight.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with detailed descriptions of changes.

## License

This project is licensed under the MIT License.

## Contact

For questions, feedback, or suggestions:

- GitHub: [SMA-Codes](https://github.com/SMA-Codes)

---
Experience the power of GPU-accelerated computing with CUDA!

