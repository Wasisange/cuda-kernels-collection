__global__ void add_vectors_cuda(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to launch the kernel
extern "C" void add_vectors_host(float *a, float *b, float *c, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add_vectors_cuda<<<numBlocks, blockSize>>>(a, b, c, n);
    cudaDeviceSynchronize(); // Wait for the GPU to finish
}
