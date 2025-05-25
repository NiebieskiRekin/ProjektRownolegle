// #include <gtest/gtest.h>
// #include "md5_cuda.cuh"

// TEST(MD5CudaTest, MD5TestEmpty) {
//     // Hashes empty string and stores signature
//     void* sig = hash_cuda("");
//     ASSERT_EQ(sig2hex(sig).compare("d41d8cd98f00b204e9800998ecf8427e"), 0);
// }


// TEST(MD5CudaTest, MD5TestText) {
//     // Hashes "The quick brown fox jumps over the lazy dog" and stores signature
//     void* sig2 =hash_cuda("The quick brown fox jumps over the lazy dog");
//     // Test with cassert whether sig is correct from the expected value
//     ASSERT_EQ(sig2hex(sig2).compare("9e107d9d372bb6826bd81d3542a419d6"), 0);
// }


// TEST(MD5CudaTest, MD5TestTextDot) {
//     // Hashes "The quick brown fox jumps over the lazy dog." (notice the
//     // additional period) and stores signature
//     void* sig2 = hash_cuda("The quick brown fox jumps over the lazy dog");
//     void* sig3 = hash_cuda("The quick brown fox jumps over the lazy dog.");
//     // Test with cassert whether sig is correct from the expected value
//     ASSERT_EQ(sig2hex(sig3).compare("e4d909c290d0fb1ca068ffaddf22cbd0"), 0);
//     ASSERT_NE(sig2hex(sig3).compare(sig2hex(sig2)), 0);
// }


// TEST(MD5CudaTest, MD5TestAlphanumeric) { 
//     // Hashes "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
//     // and stores signature
//     void* sig4 = hash_cuda("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
//     // Test with cassert whether sig is correct from the expected value
//     ASSERT_EQ(sig2hex(sig4).compare("d174ab98d277d9f5a5611c2c9f419d9f"), 0);
// }