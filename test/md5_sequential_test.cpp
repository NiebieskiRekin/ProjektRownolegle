#include <gtest/gtest.h>
#include "utils.hpp"
#include "md5_sequential.hpp"


TEST(MD5Test, MD5Test1) {
    // Hashes empty string and stores signature
    void* sig = hash("");
    // Test with cassert whether sig is correct from the expected value
    assert(sig2hex(sig).compare("d41d8cd98f00b204e9800998ecf8427e") == 0);
 
    // Hashes "The quick brown fox jumps over the lazy dog" and stores signature
    void* sig2 =hash("The quick brown fox jumps over the lazy dog");
    // Test with cassert whether sig is correct from the expected value
    assert(sig2hex(sig2).compare("9e107d9d372bb6826bd81d3542a419d6") == 0);
 
    // Hashes "The quick brown fox jumps over the lazy dog." (notice the
    // additional period) and stores signature
    void* sig3 = hash("The quick brown fox jumps over the lazy dog.");
    // Test with cassert whether sig is correct from the expected value
    assert(sig2hex(sig3).compare("e4d909c290d0fb1ca068ffaddf22cbd0") == 0);
 
    // Hashes "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    // and stores signature
    void* sig4 = hash("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
    // Test with cassert whether sig is correct from the expected value
    assert(sig2hex(sig4).compare("d174ab98d277d9f5a5611c2c9f419d9f") == 0);
}