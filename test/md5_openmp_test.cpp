#include <gtest/gtest.h>
#include "utils.hpp"
#include "md5_openmp.hpp"

TEST(MD5OmpTest, MD5TestEmpty) {
    // Hashes empty string and stores signature
    void* sig = hash_openmp("");
    auto s = sig2hex(sig);
    ASSERT_EQ(s.compare("d41d8cd98f00b204e9800998ecf8427e"), 0);
}


TEST(MD5OmpTest, MD5TestText) {
    // Hashes "The quick brown fox jumps over the lazy dog" and stores signature
    void* sig2 =hash_openmp("The quick brown fox jumps over the lazy dog");
    // Test with cassert whether sig is correct from the expected value
    ASSERT_EQ(sig2hex(sig2).compare("9e107d9d372bb6826bd81d3542a419d6"), 0);
}


TEST(MD5OmpTest, MD5TestTextDot) {
    // Hashes "The quick brown fox jumps over the lazy dog." (notice the
    // additional period) and stores signature
    void* sig2 = hash_openmp("The quick brown fox jumps over the lazy dog");
    void* sig3 = hash_openmp("The quick brown fox jumps over the lazy dog.");
    // Test with cassert whether sig is correct from the expected value
    ASSERT_EQ(sig2hex(sig3).compare("e4d909c290d0fb1ca068ffaddf22cbd0"), 0);
    ASSERT_NE(sig2hex(sig3).compare(sig2hex(sig2)), 0);
}


TEST(MD5OmpTest, MD5TestAlphanumeric) { 
    // Hashes "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    // and stores signature
    void* sig4 = hash_openmp("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    // Test with cassert whether sig is correct from the expected value
    ASSERT_EQ(sig2hex(sig4),"437bba8e0bf58337674f4539e75186ac");
}