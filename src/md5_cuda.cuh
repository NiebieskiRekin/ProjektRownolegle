#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <array>

#include "utils.hpp"

std::array<uint8_t, 16> hash_cuda(const void* input_bs, uint64_t input_size);
std::array<uint8_t, 16> hash_cuda(const std::string& message);
std::array<uint8_t, 16> hash_cuda(const void* input_bs, uint64_t input_size, int threadsPerBlock);