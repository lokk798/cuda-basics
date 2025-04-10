#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA error checking macro
#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
            exit(code);
    }
}

// CUDA Kernel
// __global__: defines a CUDA kernel, tell the compiler
// to run this function on a GPU in parallel multiple threads.
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    int N = 10000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = sinf(i) * sinf(i);
        h_b[i] = cosf(i) * cosf(i);
    }

    // Debug print first few initialized values
    std::cout << "h_a[0]: " << h_a[0] << ", h_b[0]: " << h_b[0] << std::endl;
    std::cout << "Expected h_a[0] + h_b[0] = " << h_a[0] + h_b[0] << std::endl;

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaCheckError(cudaMalloc(&d_a, size));
    cudaCheckError(cudaMalloc(&d_b, size));
    cudaCheckError(cudaMalloc(&d_c, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Configure kernel launch
    int THREADS_PER_BLOCK = 256;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Check kernel launch error and sync
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5)
        {
            std::cerr << "Mismatch at index " << i << ": " << h_c[i] << " != " << expected << std::endl;
            success = false;
            break;
        }
    }

    if (success)
    {
        std::cout << "Vector addition completed successfully!" << std::endl;
    }

    // Free device memory
    cudaCheckError(cudaFree(d_a));
    cudaCheckError(cudaFree(d_b));
    cudaCheckError(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
