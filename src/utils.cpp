#include "utils.hpp"
#include <algorithm>

const std::string characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
std::random_device random_device;
std::mt19937 generator(random_device());
std::uniform_int_distribution<> distribution(0, characters.size() - 1);

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
std::string sig2hex(const uint8_t *sig)
{
    const char *hexChars = "0123456789abcdef";
    std::string hex = "";
    for (uint8_t i = 0; i < 16; i++)
    {
        hex.push_back(hexChars[(sig[i] >> 4) & 0xF]);
        hex.push_back(hexChars[(sig[i]) & 0xF]);
    }
    return hex;
}
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

    return padded_message;
}

std::array<uint8_t, 16> build_signature(uint32_t a0, uint32_t b0, uint32_t c0, uint32_t d0){
    // Build signature from the final state
    std::array<uint8_t, 16> sig{};
    for (uint8_t i = 0; i < 4; i++){
        sig[i] = (a0 >> (8 * i)) & 0xFF;
        sig[i + 4] = (b0 >> (8 * i)) & 0xFF;
        sig[i + 8] = (c0 >> (8 * i)) & 0xFF;
        sig[i + 12] = (d0 >> (8 * i)) & 0xFF;
    }
    return sig;
}

std::string generate_random_string(size_t length) {
    std::string random_string(characters);
    std::shuffle(random_string.begin(), random_string.end(), generator);

    return random_string.substr(0, length);
}