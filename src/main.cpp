#include <iostream>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <chrono>

// #include "md5_cuda.cuh"
#include "md5_openmp.hpp"
#include "md5_sequential.hpp"

size_t get_file_size(std::ifstream& file){
    std::streampos begin,end;
    file.seekg(0, std::ios::end);
    end = file.tellg();
    file.seekg(0, std::ios::beg);
    begin = file.tellg();
    return (end-begin);
}

int main(int argc, char** argv) {
    if (argc < 2){
        std::cerr << "Za mało argumentów, użycie: main <CUDA|OPENMP|SEQUENTIAL> <pliki do przeliczenia>" << std::endl;
        exit(1);
    }

    std::string mode = argv[1];
    std::vector<std::string> files;
    std::vector<std::array<uint8_t,16>> hashes;
    std::vector<std::chrono::duration<double>> times;
    files.reserve(argc-2);
    hashes.reserve(argc-2);
    times.reserve(argc-2);
    for (int i=2; i<argc; i++){
        files.push_back(argv[i]);
    }

    if (mode == "CUDA"){
        exit(1);
        // for (const auto& filename : files){
            // std::ifstream file(filename,std::ios::in|std::ios::binary|std::ios::ate);
            // if (!file.is_open()){
            //     std::cerr << "Błąd otwarcia pliku: " << filename << std::endl;
            //     exit(1);
            // }
            // size_t data_len = get_file_size(file);
            // auto data = new char[data_len];
            // file.read(data,data_len);        
            // file.close();
            // const auto start{std::chrono::high_resolution_clock::now()};
            // hashes.push_back(hash_cuda(data,data_len));
            // const auto finish{std::chrono::high_resolution_clock::now()};
            // delete[] data;
            // times.push_back(finish-start);
        // }
    } else if (mode == "OPENMP") {
        for (const auto& filename : files){
            std::ifstream file(filename,std::ios::in|std::ios::binary|std::ios::ate);
            if (!file.is_open()){
                std::cerr << "Błąd otwarcia pliku: " << filename << std::endl;
                exit(1);
            }
            size_t data_len = get_file_size(file);
            auto data = new char[data_len];
            file.read(data,data_len);        
            file.close();
            const auto start{std::chrono::high_resolution_clock::now()};
            hashes.push_back(hash_openmp(data,data_len));
            const auto finish{std::chrono::high_resolution_clock::now()};
            delete[] data;
            times.push_back(finish-start);
        }
    } else {
        for (const auto& filename : files){
            std::ifstream file(filename,std::ios::in|std::ios::binary|std::ios::ate);
            if (!file.is_open()){
                std::cerr << "Błąd otwarcia pliku: " << filename << std::endl;
                exit(1);
            }
            size_t data_len = get_file_size(file);
            auto data = new char[data_len];
            file.read(data,data_len);        
            file.close();
            const auto start{std::chrono::high_resolution_clock::now()};
            hashes.push_back(hash_sequential(data,data_len));
            const auto finish{std::chrono::high_resolution_clock::now()};
            delete[] data;
            times.push_back(finish-start);
        }
    }


    for (size_t i=0; i<files.size(); i++){
        auto hash_ptr = static_cast<const uint8_t*>(hashes[i].data());
        const auto hash_hex = sig2hex(hash_ptr);

        std::cout << files[i] << ", " << times[i].count() << ": " << hash_hex << std::endl;
    }
    // std::cout << std::endl;

    return 0;
}