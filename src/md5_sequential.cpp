#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "utils.hpp"

std::vector<uint8_t> preprocess(const uint8_t *input, uint64_t input_size){
    // Processing the bytestring
    // First compute the size the padded message will have
    // so it is possible to allocate the right amount of memory
    uint64_t padded_message_size = 0;
    if (input_size % 64 < 56){
        padded_message_size = input_size + 64 - (input_size % 64);
    } else{
        padded_message_size = input_size + 128 - (input_size % 64);
    }

    std::vector<uint8_t> padded_message(padded_message_size);

    // Beginning of the padded message is the original message
    std::copy(input, input + input_size, padded_message.begin());

    // Afterwards comes a single 1 bit and then only zeroes
    padded_message[input_size] = 1 << 7; // 10000000
    for (uint64_t i = input_size; i % 64 != 56; i++) {
        if (i == input_size){
            continue; // pass first iteration
        }
        padded_message[i] = 0;
    }

    // Add the 64-bit size of the message at the end in LittleEndian
    uint64_t input_bitsize_le = toLittleEndian64(input_size * 8);
    for (uint8_t i = 0; i < 8; i++){
        padded_message[padded_message.size() - 8 + i] = (input_bitsize_le >> (56 - 8 * i)) & 0xFF;
    }

    return std::move(padded_message);
}


void processChunk(const uint8_t *padded_message, uint64_t chunk_start, uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0){
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

uint8_t * build_signature(uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0){
    // Build signature from the final state
    auto *sig = new uint8_t[16];
    for (uint8_t i = 0; i < 4; i++){
        sig[i] = (a0 >> (8 * i)) & 0xFF;
        sig[i + 4] = (b0 >> (8 * i)) & 0xFF;
        sig[i + 8] = (c0 >> (8 * i)) & 0xFF;
        sig[i + 12] = (d0 >> (8 * i)) & 0xFF;
    }
    return sig;
}

void *hash(const void *input_bs, uint64_t input_size){

    auto *input = static_cast<const uint8_t *>(input_bs);

    std::array<uint32_t, 4> state = initial_128_bit_state;

    auto padded_message = preprocess(input,input_size);

    // Rounds
    for (uint64_t chunk = 0; chunk * 64 < padded_message.size(); chunk++){
        processChunk(padded_message.data(), chunk * 64, state[0], state[1], state[2], state[3]);
    }

    auto sig = build_signature(state[0], state[1], state[2], state[3]);
    return sig;
}

void *hash(const std::string &message){
    return hash(&message[0], message.size());
}

