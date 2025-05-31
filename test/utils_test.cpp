#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "utils.hpp"


TEST(LeftRotate32bitsTest, BasicRotation) {
    uint32_t num = 0x12345678;
    ASSERT_EQ(leftRotate32bits(num, 4), 0x23456781);
}

TEST(LeftRotate32bitsTest, FullRotation) {
    uint32_t num = 0xABCDEF01;
    ASSERT_EQ(leftRotate32bits(num, 32), 0xABCDEF01);
}

TEST(LeftRotate32bitsTest, ZeroRotation) {
    uint32_t num = 0xF0E1D2C3;
    ASSERT_EQ(leftRotate32bits(num, 0), 0xF0E1D2C3);
}

TEST(LeftRotate32bitsTest, LargeRotation) {
    uint32_t num = 0x98765432;
    ASSERT_EQ(leftRotate32bits(num, 36), 0x87654329);
}

TEST(IsBigEndianTest, ReturnsBool) {
    ASSERT_TRUE(isBigEndian() || !isBigEndian());
}

TEST(ToLittleEndian32Test, LittleEndianInput) {
    if (!isBigEndian()) {
        uint32_t num = 0x04030201;
        ASSERT_EQ(toLittleEndian32(num), 0x01020304);
    }
}

TEST(ToLittleEndian32Test, BigEndianInput) {
    if (isBigEndian()) {
        uint32_t num = 0x01020304;
        ASSERT_EQ(toLittleEndian32(num), 0x04030201);
    }
}

TEST(ToLittleEndian32Test, SingleByte) {
    uint32_t num = 0x000000FF;
    uint32_t expected = isBigEndian() ? 0x000000FF : 0xFF000000;
    ASSERT_EQ(toLittleEndian32(num), expected);
}

TEST(ToLittleEndian64Test, LittleEndianInput) {
    if (!isBigEndian()) {
        uint64_t num = 0x0807060504030201;
        ASSERT_EQ(toLittleEndian64(num), 0x0102030405060708);
    }
}

TEST(ToLittleEndian64Test, BigEndianInput) {
    if (isBigEndian()) {
        uint64_t num = 0x0102030405060708;
        ASSERT_EQ(toLittleEndian64(num), 0x0807060504030201);
    }
}

TEST(Sig2HexTest, BasicConversion) {
    std::array<uint8_t, 16> signature = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
    ASSERT_EQ(sig2hex(signature.data()), "0123456789abcdeffedcba9876543210");
}

TEST(Sig2HexTest, AllZeros) {
    std::array<uint8_t, 16> signature = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    ASSERT_EQ(sig2hex(signature.data()), "00000000000000000000000000000000");
}

TEST(Sig2HexTest, AllOnes) {
    std::array<uint8_t, 16> signature = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    ASSERT_EQ(sig2hex(signature.data()), "ffffffffffffffffffffffffffffffff");
}