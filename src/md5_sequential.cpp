#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "utils.hpp"
#include "md5_sequential.hpp"

void process_chunk_sequential(const uint8_t *padded_message, uint64_t chunk_start, uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0){
    std::array<uint32_t, 16> blocks{};
    uint32_t A = a0;
    uint32_t B = b0;
    uint32_t C = c0;
    uint32_t D = d0;

    // First, build the 16 32-bits blocks from the chunk
    for (uint8_t bid = 0; bid < 16; bid++){
        blocks[bid] = 0;
        for (uint8_t cid = 0; cid < 4; cid++){
            blocks[bid] = (blocks[bid] << 8) + padded_message[chunk_start + bid * 4 + cid];
        }
    }

    // Main "hashing" loop
    for (uint8_t i = 0; i < 64; i++){
        uint32_t F = 0, g = 0;
        if (i < 16){
            F = (B & C) | ((~B) & D);
            g = i;
        }
        else if (i < 32){
            F = (D & B) | ((~D) & C);
            g = (5 * i + 1) % 16;
        }
        else if (i < 48){
            F = B ^ C ^ D;
            g = (3 * i + 5) % 16;
        }
        else{
            F = C ^ (B | (~D));
            g = (7 * i) % 16;
        }

        // Update the accumulators
        F += A + K[i] + toLittleEndian32(blocks[g]);

        A = D;
        D = C;
        C = B;
        B += leftRotate32bits(F, s[i]);
    }
    // Update the state with this chunk's hash
    a0 += A;
    b0 += B;
    c0 += C;
    d0 += D;
}



std::array<uint8_t, 16> hash_sequential(const void *input_bs, uint64_t input_size){

    auto *input = static_cast<const uint8_t *>(input_bs);

    std::array<uint32_t, 4> state = initial_128_bit_state;

    auto padded_message = preprocess(input,input_size);

    // Rounds
    for (uint64_t chunk = 0; chunk * 64 < padded_message.size(); chunk++){
        process_chunk_sequential(padded_message.data(), chunk * 64, state[0], state[1], state[2], state[3]);
    }

    auto sig = build_signature(state[0], state[1], state[2], state[3]);
    return sig;
}

std::array<uint8_t, 16> hash_sequential(const std::string &message){
    return hash_sequential(&message[0], message.size());
}

