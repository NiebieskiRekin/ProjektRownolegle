// #include <benchmark/benchmark.h>
// #include <iostream>
// #include <string>
// #include <vector>

// #include "utils.hpp"
// #include "md5_openmp.hpp"

// static void BM_OpenmpMD5_Small(benchmark::State& state) {
//     std::string message(state.range(0), 'a');
//     for (auto _ : state) {
//         void* result = hash_openmp(message);
//         benchmark::DoNotOptimize(result);
//         delete[] static_cast<uint8_t*>(result);
//     }
//     state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
// }
// BENCHMARK(BM_OpenmpMD5_Small)->RangeMultiplier(2)->Range(8, 8 * 1024);

// static void BM_OpenmpMD5_Medium(benchmark::State& state) {
//     std::string message(state.range(0), 'b');
//     for (auto _ : state) {
//         void* result = hash_openmp(message);
//         benchmark::DoNotOptimize(result);
//         delete[] static_cast<uint8_t*>(result);
//     }
//     state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
// }
// BENCHMARK(BM_OpenmpMD5_Medium)->RangeMultiplier(4)->Range(16 * 1024, 256 * 1024);

// static void BM_OpenmpMD5_Large(benchmark::State& state) {
//     std::string message(state.range(0), 'c');
//     for (auto _ : state) {
//         void* result = hash_openmp(message);
//         benchmark::DoNotOptimize(result);
//         delete[] static_cast<uint8_t*>(result);
//     }
//     state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
// }
// BENCHMARK(BM_OpenmpMD5_Large)->RangeMultiplier(8)->Range(512 * 1024, 4 * 1024 * 1024);