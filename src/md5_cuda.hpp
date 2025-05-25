#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <array>

#include "utils.hpp"
#include "md5_kernels.cuh"

uint8_t* hash_cuda(const void* input_bs, uint64_t input_size);
uint8_t* hash_cuda(const std::string& message);