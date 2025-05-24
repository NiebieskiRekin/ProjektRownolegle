#include <iostream>
#include "some_code.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

int main() {
    std::cout << "Hello from the main application!" << std::endl;
    someFunction();

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "Running with OpenMP, max threads: " << num_threads << std::endl;
#else
    std::cout << "OpenMP not enabled." << std::endl;
#endif

#ifdef __CUDACC__
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id == cudaSuccess) {
        std::cout << "CUDA devices found: " << deviceCount << std::endl;
    } else {
        std::cerr << "CUDA error getting device count: " << cudaGetErrorString(error_id) << std::endl;
    }
#else
    std::cout << "CUDA not enabled." << std::endl;
#endif

    return 0;
}