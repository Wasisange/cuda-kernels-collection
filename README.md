# CUDA Kernels Collection

Custom CUDA kernels for optimized tensor operations in deep learning.

## Overview
This repository contains a collection of custom CUDA kernels designed to accelerate specific tensor operations commonly found in deep learning workloads. These kernels are optimized for NVIDIA GPUs and can be integrated into PyTorch, TensorFlow, or other deep learning frameworks via custom C++/CUDA extensions.

## Features
- Optimized matrix multiplication (GEMM) kernels.
- Custom activation functions.
- Efficient data loading and preprocessing kernels.
- Example integrations with PyTorch.

## Installation

```bash
git clone https://github.com/Wasisange/cuda-kernels-collection.git
cd cuda-kernels-collection
# Compile kernels (example for PyTorch extension)
pip install torch
python setup.py install
```

## Usage

```cpp
// Example of a custom CUDA kernel
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host-side wrapper
void add_gpu(float* a, float* b, float* c, int n) {
    // ... CUDA setup and kernel launch ...
}
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
