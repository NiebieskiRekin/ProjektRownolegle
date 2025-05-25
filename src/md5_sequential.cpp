#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "utils.hpp"



void *hash_bs(const void *input_bs, uint64_t input_size){

    auto *input = static_cast<const uint8_t *>(input_bs);

    // The initial 128-bit state
    uint32_t a0 = 0x67452301, A = 0;
    uint32_t b0 = 0xefcdab89, B = 0;
    uint32_t c0 = 0x98badcfe, C = 0;
    uint32_t d0 = 0x10325476, D = 0;

    // Step 1: Processing the bytestring
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

    // We then have to add the 64-bit size of the message at the end
    // When there is a conversion from int to bytestring or vice-versa
    // We always need to make sure it is little endian
    uint64_t input_bitsize_le = toLittleEndian64(input_size * 8);

    for (uint8_t i = 0; i < 8; i++){
        padded_message[padded_message_size - 8 + i] = (input_bitsize_le >> (56 - 8 * i)) & 0xFF;
    }

    // Already allocate memory for blocks
    std::array<uint32_t, 16> blocks{};

    // Rounds
    for (uint64_t chunk = 0; chunk * 64 < padded_message_size; chunk++){
        // First, build the 16 32-bits blocks from the chunk
        for (uint8_t bid = 0; bid < 16; bid++){
            blocks[bid] = 0;
            // Having to build a 32-bit word from 4-bit words
            // Add each and shift them to the left
            for (uint8_t cid = 0; cid < 4; cid++){
                blocks[bid] = (blocks[bid] << 8) + padded_message[chunk * 64 + bid * 4 + cid];
            }
        }

        A = a0;
        B = b0;
        C = c0;
        D = d0;

        // Main "hashing" loop
        for (uint8_t i = 0; i < 64; i++){
            uint32_t F = 0, g = 0;
            if (i < 16){
                F = (B & C) | ((~B) & D);
                g = i;
            } else if (i < 32){
                F = (D & B) | ((~D) & C);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                F = B ^ C ^ D;
                g = (3 * i + 5) % 16;
            } else {
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
    auto *sig = new uint8_t[16];

    for (uint8_t i = 0; i < 4; i++){
        sig[i] = (a0 >> (8 * i)) & 0xFF;
        sig[i + 4] = (b0 >> (8 * i)) & 0xFF;
        sig[i + 8] = (c0 >> (8 * i)) & 0xFF;
        sig[i + 12] = (d0 >> (8 * i)) & 0xFF;
    }
    return sig;
}

void *hash(const std::string &message){
    return hash_bs(&message[0], message.size());
}

