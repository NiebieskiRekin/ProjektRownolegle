#include <algorithm>  
#include <array>        
#include <cstdint>
#include <iostream>   
#include <string>     
#include <vector>     
 
uint32_t leftRotate32bits(uint32_t n, std::size_t rotate);
bool isBigEndian();
uint32_t toLittleEndian32(uint32_t n);
uint64_t toLittleEndian64(uint64_t n);
std::string sig2hex(void* sig);
void* hash_bs(const void* input_bs, uint64_t input_size);
void* hash(const std::string& message);
