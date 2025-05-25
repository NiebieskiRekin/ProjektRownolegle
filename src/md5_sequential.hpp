#pragma once
#include "utils.hpp"

void *hash_bs(const void *input_bs, uint64_t input_size);
void *hash(const std::string &message);
std::vector<uint8_t> preprocess(const uint8_t *input, uint64_t input_size);
void processChunk(const uint8_t *padded_message, uint64_t chunk_start, uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0);
uint8_t * build_signature(uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0);

