#include <gtest/gtest.h>
#include "utils.hpp"
#include "md5_sequential.hpp"


TEST(MD5SequentialTest, MD5TestEmpty) {
    auto res = hash_sequential("");
    auto sig = static_cast<uint8_t*>(res.data());
    ASSERT_EQ(sig2hex(sig).compare("d41d8cd98f00b204e9800998ecf8427e"), 0);
}


TEST(MD5SequentialTest, MD5TestText) {
    auto res = hash_sequential("The quick brown fox jumps over the lazy dog");
    auto sig = static_cast<uint8_t*>(res.data());
    ASSERT_EQ(sig2hex(sig).compare("9e107d9d372bb6826bd81d3542a419d6"), 0);
}


TEST(MD5SequentialTest, MD5TestTextDot) {
    auto res2 = hash_sequential("The quick brown fox jumps over the lazy dog");
    auto res3 = hash_sequential("The quick brown fox jumps over the lazy dog.");
    auto sig2 = static_cast<uint8_t*>(res2.data());
    auto sig3 = static_cast<uint8_t*>(res3.data());
    ASSERT_EQ(sig2hex(sig3).compare("e4d909c290d0fb1ca068ffaddf22cbd0"), 0);
    ASSERT_NE(sig2hex(sig3).compare(sig2hex(sig2)), 0);
}


TEST(MD5SequentialTest, MD5TestAlphanumeric) { 
    auto res = hash_sequential("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
    auto sig4 = static_cast<uint8_t*>(res.data());
    ASSERT_EQ(sig2hex(sig4).compare("d174ab98d277d9f5a5611c2c9f419d9f"), 0);
}