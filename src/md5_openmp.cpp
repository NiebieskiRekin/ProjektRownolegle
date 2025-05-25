#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <cmath>

#include "utils.hpp"
#include "md5_sequential.hpp"

std::array<uint32_t, 4> combineStates(const std::array<uint32_t, 4>& state1, const std::array<uint32_t, 4>& state2) {
    std::vector<uint8_t> combined_input(32);

    // Serialize state1 to bytes (little endian)
    for (uint8_t i = 0; i < 4; ++i) {
        combined_input[i * 4 + 0] = (state1[i] >> 0) & 0xFF;
        combined_input[i * 4 + 1] = (state1[i] >> 8) & 0xFF;
        combined_input[i * 4 + 2] = (state1[i] >> 16) & 0xFF;
        combined_input[i * 4 + 3] = (state1[i] >> 24) & 0xFF;
    }

    // Serialize state2 to bytes (little endian)
    for (uint8_t i = 0; i < 4; ++i) {
        combined_input[16 + i * 4 + 0] = (state2[i] >> 0) & 0xFF;
        combined_input[16 + i * 4 + 1] = (state2[i] >> 8) & 0xFF;
        combined_input[16 + i * 4 + 2] = (state2[i] >> 16) & 0xFF;
        combined_input[16 + i * 4 + 3] = (state2[i] >> 24) & 0xFF;
    }

    // Hash the combined input using MD5
    void* hash_result = hash_sequential(combined_input.data(), 32);
    auto* hash_bytes = static_cast<uint8_t*>(hash_result);

    // Convert the resulting 16-byte MD5 digest back to a 4-element uint32_t array (little endian)
    std::array<uint32_t, 4> combined_state;
    for (uint8_t i = 0; i < 4; ++i) {
        combined_state[i] = (static_cast<uint32_t>(hash_bytes[i * 4 + 0]) << 0) |
                             (static_cast<uint32_t>(hash_bytes[i * 4 + 1]) << 8) |
                             (static_cast<uint32_t>(hash_bytes[i * 4 + 2]) << 16) |
                             (static_cast<uint32_t>(hash_bytes[i * 4 + 3]) << 24);
    }

    delete[] hash_bytes;
    return combined_state;
}

void *hash_openmp(const void *input_bs, uint64_t input_size){
    auto *input = static_cast<const uint8_t *>(input_bs);

    // The initial 128-bit state
    std::array<uint32_t, 4> original_state = initial_128_bit_state;

    auto padded_message = preprocess(input,input_size);

    uint64_t num_chunks = padded_message.size() / 64;
    std::vector<std::array<uint32_t, 4>> chunk_states(num_chunks, original_state);

#pragma omp parallel for
    for (uint64_t i = 0; i < num_chunks; i++){
        process_chunk_sequential(padded_message.data(), i * 64, chunk_states[i][0], chunk_states[i][1], chunk_states[i][2], chunk_states[i][3]);
    }

    // Merkle tree-like reduction of chunk states
    uint64_t num_levels = std::ceil(std::log2(num_chunks));
    std::vector<std::vector<std::array<uint32_t, 4>>> levels(num_levels + 1);
    levels[0] = chunk_states;

    for (uint64_t level = 1; level <= num_levels; ++level) {
        uint64_t num_nodes_at_level = levels[level - 1].size();
        for (uint64_t i = 0; i < num_nodes_at_level; i += 2) {
            if (i + 1 < num_nodes_at_level) {
                levels[level].push_back(combineStates(levels[level - 1][i], levels[level - 1][i + 1]));
            } else {
                levels[level].push_back(levels[level - 1][i]); // If odd number of nodes, carry the last one up
            }
        }
    }

    std::array<uint32_t, 4> final_state = levels[num_levels][0];

    // Build signature from the final state
    auto *sig = build_signature(final_state[0],final_state[1],final_state[2],final_state[3]);
    return sig;
}

void *hash_openmp(const std::string &message)
{
    return hash_openmp(&message[0], message.size());
}

