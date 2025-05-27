#include <iostream>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

// #include "md5_cuda.cuh"
// #include "md5_kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 2){
        std::cerr << "Za mało argumentów, użycie: main <CUDA|OPENCV|SEQUENTIAL> <pliki do przeliczenia>" << std::endl;
        exit(1);
    }

    std::string mode = argv[1];
    std::vector<std::string> files;
    files.reserve(argc-2);
    for (int i=2; i<argc; i++){
        files.push_back(argv[i]);
    }

    if (mode == "CUDA"){
        std::cout << "CUDA: " << std::endl;
    } else if (mode == "OPENCV") {
        std::cout << "OPENCV: " << std::endl;
    } else {
        std::cout << "SEQUENTIAL: " << std::endl;
    }

    for (const auto& file : files){
        std::cout << file << " ";
    }
    std::cout << std::endl;

    return 0;
}