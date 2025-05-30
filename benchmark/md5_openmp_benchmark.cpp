#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <vector>

#include "utils.hpp"
#include "md5_openmp.hpp"

static void BM_OpenmpMD5(benchmark::State& state) {
    std::string message(state.range(0), 'd');
    for (auto _ : state) {
        auto result = hash_openmp(message);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
}
BENCHMARK(BM_OpenmpMD5)->Range(1024, 2*1024UL * 1024UL * 1024UL);