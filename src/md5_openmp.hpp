#pragma once
#include "utils.hpp"

std::array<uint8_t, 16> hash_openmp(const void *input_bs, uint64_t input_size);
std::array<uint8_t, 16> hash_openmp(const std::string &message);
void merkleTreeReduce(std::vector<std::array<uint32_t, 4>>& chunk_states);