#pragma once
#include "utils.hpp"

std::array<uint8_t, 16> hash_sequential(const void *input_bs, uint64_t input_size);
std::array<uint8_t, 16> hash_sequential(const std::string &message);
void process_chunk_sequential(const uint8_t *padded_message, uint64_t chunk_start, uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0);

