#include "utils.hpp"

uint32_t leftRotate32bits(uint32_t n, std::size_t rotate)
{
    return (n << rotate) | (n >> (32 - rotate));
}
bool isBigEndian()
{
    union
    {
        uint32_t i;
        std::array<char, 4> c;
    } bint = {0x01020304};

    return bint.c[0] == 1;
}
uint32_t toLittleEndian32(uint32_t n)
{
    if (!isBigEndian())
    {
        return ((n << 24) & 0xFF000000) | ((n << 8) & 0x00FF0000) |
               ((n >> 8) & 0x0000FF00) | ((n >> 24) & 0x000000FF);
    }
    return n;
}
uint64_t toLittleEndian64(uint64_t n)
{
    if (!isBigEndian())
    {
        return ((n << 56) & 0xFF00000000000000) |
               ((n << 40) & 0x00FF000000000000) |
               ((n << 24) & 0x0000FF0000000000) |
               ((n << 8) & 0x000000FF00000000) |
               ((n >> 8) & 0x00000000FF000000) |
               ((n >> 24) & 0x0000000000FF0000) |
               ((n >> 40) & 0x000000000000FF00) |
               ((n >> 56) & 0x00000000000000FF);
    }
    return n;
}
std::string sig2hex(void *sig)
{
    const char *hexChars = "0123456789abcdef";
    auto *intsig = static_cast<uint8_t *>(sig);
    std::string hex = "";
    for (uint8_t i = 0; i < 16; i++)
    {
        hex.push_back(hexChars[(intsig[i] >> 4) & 0xF]);
        hex.push_back(hexChars[(intsig[i]) & 0xF]);
    }
    return hex;
}


void processChunk(const uint8_t *padded_message, uint64_t chunk_start, const std::array<uint32_t, 64> &s, const std::array<uint32_t, 64> &K, uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0){
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