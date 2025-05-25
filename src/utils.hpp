#pragma once
#include <cstdint>
#include <array>
#include <string>

uint32_t leftRotate32bits(uint32_t n, std::size_t rotate);
bool isBigEndian();
uint32_t toLittleEndian32(uint32_t n);
uint64_t toLittleEndian64(uint64_t n);
std::string sig2hex(void *sig);
void processChunk(const uint8_t *padded_message, uint64_t chunk_start, const std::array<uint32_t, 64> &s, const std::array<uint32_t, 64> &K, uint32_t &a0, uint32_t &b0, uint32_t &c0, uint32_t &d0);