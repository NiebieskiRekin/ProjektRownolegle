#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <array>

#include "utils.hpp"
#include "md5_kernels.cuh"

std::array<uint8_t, 16> hash_cuda(const void* input_bs, uint64_t input_size);
std::array<uint8_t, 16> hash_cuda(const std::string& message);