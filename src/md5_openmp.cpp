#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "utils.hpp"

void *hash_bs_openmp(const void *input_bs, uint64_t input_size){
    auto *input = static_cast<const uint8_t *>(input_bs);

    // The initial 128-bit state
    std::array<uint32_t, 4> state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    std::array<uint32_t, 4> original_state = state;

    // Step 1: Processing the bytestring

    uint64_t padded_message_size = input_size;
    if (input_size % 64 < 56){
        padded_message_size = input_size + 64 - (input_size % 64);
    } else{
        padded_message_size = input_size + 128 - (input_size % 64);
    }

    std::vector<uint8_t> padded_message(padded_message_size);
    std::copy(input, input + input_size, padded_message.begin());
    padded_message[input_size] = 1 << 7;
    for (uint64_t i = input_size + 1; i % 64 != 56; ++i){
        padded_message[i] = 0;
    }
    uint64_t input_bitsize_le = toLittleEndian64(input_size * 8);
    for (uint8_t i = 0; i < 8; ++i){
        padded_message[padded_message_size - 8 + i] = (input_bitsize_le >> (56 - 8 * i)) & 0xFF;
    }

    uint64_t num_chunks = padded_message_size / 64;
    std::vector<std::array<uint32_t, 4>> chunk_states(num_chunks, original_state);

#pragma omp parallel for
    for (uint64_t chunk = 0; chunk < num_chunks; ++chunk){
        uint32_t a0 = chunk_states[chunk][0];
        uint32_t b0 = chunk_states[chunk][1];
        uint32_t c0 = chunk_states[chunk][2];
        uint32_t d0 = chunk_states[chunk][3];
        processChunk(padded_message.data(), chunk * 64, s, K, a0, b0, c0, d0);
        chunk_states[chunk] = {a0, b0, c0, d0};
    }

    // Combine the results (in MD5, this is a sequential accumulation)
    for (auto &state : chunk_states){
        state[0] += original_state[0];
        state[1] += original_state[1];
        state[2] += original_state[2];
        state[3] += original_state[3];
        original_state = state; // Accumulate for the next chunk (though not strictly parallelizable this way for final result)
    }

    // Build signature from the final state
    auto *sig = new uint8_t[16];
    for (uint8_t i = 0; i < 4; ++i){
        sig[i] = (original_state[0] >> (8 * i)) & 0xFF;
        sig[i + 4] = (original_state[1] >> (8 * i)) & 0xFF;
        sig[i + 8] = (original_state[2] >> (8 * i)) & 0xFF;
        sig[i + 12] = (original_state[3] >> (8 * i)) & 0xFF;
    }

    return sig;
}

void *hash_openmp(const std::string &message)
{
    return hash_bs_openmp(&message[0], message.size());
}

