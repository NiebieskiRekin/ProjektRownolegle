#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <cmath>

#include "utils.hpp"
#include "md5_sequential.hpp"

void combineStates(const std::array<uint32_t, 4>& state1, const std::array<uint32_t, 4>& state2, std::array<uint32_t, 4>& combined_state) {
    combined_state[0] = state1[0] + state2[0];
    combined_state[1] = state1[1] + state2[1];
    combined_state[2] = state1[2] + state2[2];
    combined_state[3] = state1[3] + state2[3];
}

std::array<uint8_t, 16> hash_openmp(const void *input_bs, uint64_t input_size){
    auto *input = static_cast<const uint8_t *>(input_bs);

    std::array<uint32_t, 4> state = initial_128_bit_state;

    auto padded_message = preprocess(input,input_size);

    uint64_t num_chunks = padded_message.size() / 64;
    std::vector<std::array<uint32_t, 4>> chunk_states(num_chunks, state);

    #pragma omp parallel for
    for (uint64_t i = 0; i < num_chunks; i++){
        process_chunk_sequential(padded_message.data(), i * 64, chunk_states[i][0], chunk_states[i][1], chunk_states[i][2], chunk_states[i][3]);
    }

    // Merkle tree-like reduction of chunk states
    uint64_t num_levels = std::ceil(std::log2(num_chunks)); // https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM9KuADJ4DJgAcj4ARpjEEgAcpAAOqAqETgwe3r7%2B0ilpjgIhYZEsMXFciXaYDhlCBEzEBFk%2BfgG2mPaFDPWNBMUR0bEJtg1NLTntCmP9oYNlw5UAlLaoXsTI7BzmAMyhyN5YANQmO25RhKfYJhoAgrv7h5gnZyxMBAgAdAhXN/d3TC8RCOtFQwDMEC8DDSwDC6COqiWJwA7FY7jcAJwmACsVlxTAUCh8mAgqiOYDApwAIkcNEscVSGac0f8McRMAR1gwjtN0CAQBcCAB9ADueHQH1JSIAtEcuMy/iZkYz0QCgagQWCzKLCAghRpIdC8LDMPDESiWZj2ZziNyydTqbSXgAxJ0gHkEPkCwg6iUIKVHWXynaW5WKtXA0HgoUKUEiw0wuEIpFKy0aNkcrker2C32S81BhXosOq26AyNaoVMasJ41JxGpv5HZtHa1ZzZ0CBRiGI%2BkhxUl/63FtHDgrWicbG8PwcLSkVCcNzWaw8tYbZ67HikAiaMcrADWEgxnwAbPFsWYNJIT1wTxiMfEMRf9JxJNPd/POLwFCANNvdyscCwEgaAsEkdCxOQlCgeB9BxMACjMEkCgIKgBB8HQBCxD%2BEBRB%2BFzMMQACenBbgRjREQA8lE2g1Du3C8KBbCCJRDC0CRs68FgUReMAbhiLQP4MaQWBvEY4icSJeDsrUABumBCXOmCqDUQJbFuoRYROkm0HgUTEBRHhYB%2BBDEHgLCkbw8nEFEqSYFSmBicAulGABfAGAhABqeCYCKlFJIwlkyIIIhiOwUjBfIShqB%2BuhcPohjGMulj6HpP6QCsqBJN0QnSry1KmJY1hmCegaUTsvCoNZZlYOlEArNUtTOBArgTH48XBHMpTlHo%2BTpAIbW9ak/UMAM3WLB0XR1DMg3xY13S9E0Y1DBUox9LNa1LV1K0SA1a6bLtL4cFOpAznOC4cAi8QntKJ6SEcwDIMgRwQKZUL7kiEC4IQJAnGYOxcEsvD0VoSwHv4ZifBoZjxGYUj3js2LxMi2JXkdb6nR%2BF3fr%2B/6cYBMCICgqBgRBZAUBAMFkygBhGFWhKxDlQpvQw%2B4YbQWHEDheGSeRxFBXzVE0XRQVMYwBCsexH7cbx/G0IJQWiYlElzvgMmOPJim8MpqlYUFmmdB%2Bun6YZGBbHOpnmUF1m2UoDlOS5oD4%2B5TBeT5fkBTOW78CFojiBFPtRSo6iSboZgJa5hVWClxt1Zl2UZLl%2BU7FSUfFaV0rlZV1Xigp8ANZ0dEZC4DDuJ4rQgDspCdSUO1cOHfXdINVeNxky0LBU4fzdN63lzkVfdwIi2zLXHcSF3M1934A8zO3PX13t6wHYDR0nWdlWcFdN13UcByJS9BJKE0idHCzH0vd9RDEH9ANA3joPg5UnwwyeSMaDsj6SFwiPYuj76SdjWwuMQZjlIEBImVM4JQUpiTWCww9500PozROzNiDvXZpzbm%2BFQgUQFjg4i1FaIOFFiTZiEs2IcVVpgHifEBJCS3ErcS5suLSSLprD8OtkBqX1oIQ2Ok9IGWIkZZh24zIWWEjbOy9tlahCdqDF2btfL%2BUCsJQOoV/bSEDooYOsVK4RySkVGOaV87zgTgIJOnoCrJQsCVMqOxAy8mNtSY2adLDZ1iDVPOGVJpF2aq1Keega7zHnskYaTcAnxVbkUbaY85qFyaj0Se2R2o%2BIScPOeE1pi92SXoLJW1R7z0XuuQ62k15Y03qoa6t17oIOAAfBmx9zGnzQazT6l9fqbjviAsGpBDyXj/pjABX4gF/m6WAwmEAkBrAIEkIE0DIGxHCKwLYlTt73Ues9V6LSPra3wFfcUeg1F%2B3Cpo2Q2iYqhz0CKAySRLLjknP/c6nBKJAlmQQI4qAqBb2qQ9J6L0z6fQ8KTOCN9AbAwAuDfp2kMbr0/BwHGoyIVHTMI8je8L757lINZNIzhJBAA
    std::vector<std::vector<std::array<uint32_t, 4>>> levels(num_levels + 1);
    levels[0] = chunk_states;

    for (uint64_t level = 1; level <= num_levels; ++level) {
        uint64_t num_nodes_at_level = levels[level - 1].size();
        for (uint64_t i = 0; i < num_nodes_at_level; i += 2) {
            if (i + 1 < num_nodes_at_level) {
                std::array<uint32_t, 4> res;
                combineStates(levels[level - 1][i], levels[level - 1][i + 1],res);
                levels[level].push_back(res);
            } else {
                levels[level].push_back(levels[level - 1][i]); // If odd number of nodes, carry the last one up
            }
        }
    }

    const auto& final_state = levels[num_levels][0];
    auto sig = build_signature(final_state[0],final_state[1],final_state[2],final_state[3]);
    return sig;
}

std::array<uint8_t, 16> hash_openmp(const std::string &message)
{
    return hash_openmp(&message[0], message.size());
}

