%%writefile gp.cu
#include <iostream>
#include <cuda_runtime.h> // For cudaError_t and cudaGetErrorString
using namespace std;

// Kernel function to add two matrices
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Calculate column index

    if (i < N && j < N) {
        int idx = i * N + j; // Flatten 2D index into 1D
        C[idx] = A[idx] + B[idx];
    }
}

void checkCudaError(const char* task) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA Error during " << task << ": " << cudaGetErrorString(error) << endl;
    }
}

int main() {
    int N = 512;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];
    float *dA, *dB, *dC;
    size_t size = N * N * sizeof(float);

    // Initialize host arrays with values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { // Corrected this line
            int idx = i * N + j; // Flatten 2D indices into 1D
            A[idx] = float(i * j);
            B[idx] = float((i * 2) * j);
        }
    }

    // Allocate device memory
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    // Copy input arrays to device
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    // Kernel invocation
    VecAdd<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    checkCudaError("kernel execution"); // Check for kernel errors

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();
    checkCudaError("synchronization");

    // Copy result back to host
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy");

    // Print results for a few elements
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = i * N + j;
            cout << "C[" << i << "][" << j << "] = " << C[idx] << endl;
        }
    }

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
