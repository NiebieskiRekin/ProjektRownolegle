#pragma once
#include "utils.hpp"

void *hash_openmp(const void *input_bs, uint64_t input_size);
void *hash_openmp(const std::string &message);