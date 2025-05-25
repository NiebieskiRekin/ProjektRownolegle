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
