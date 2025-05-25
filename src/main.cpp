#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

int main() {

    int num_threads = omp_get_max_threads();
    std::cout << "Running with OpenMP, max threads: " << num_threads << std::endl;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id == cudaSuccess) {
        std::cout << "CUDA devices found: " << deviceCount << std::endl;
    } else {
        std::cerr << "CUDA error getting device count: " << cudaGetErrorString(error_id) << std::endl;
    }

    return 0;
}