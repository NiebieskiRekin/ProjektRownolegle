#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "md5_cuda.cuh"

static void BM_CudaMD5_Tiny(benchmark::State& state) {
    std::string message(state.range(0), 'a');
    for (auto _ : state) {
        auto result = hash_cuda(message);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
}
BENCHMARK(BM_CudaMD5_Tiny)->RangeMultiplier(2)->Range(8, 8 * 1024);

static void BM_CudaMD5_Small(benchmark::State& state) {
    std::string message(state.range(0), 'b');
    for (auto _ : state) {
        auto result = hash_cuda(message);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
}
BENCHMARK(BM_CudaMD5_Small)->RangeMultiplier(4)->Range(16 * 1024, 256 * 1024);

static void BM_CudaMD5_Medium(benchmark::State& state) {
    std::string message(state.range(0), 'c');
    for (auto _ : state) {
        auto result = hash_cuda(message);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
}
BENCHMARK(BM_CudaMD5_Medium)->RangeMultiplier(8)->Range(512 * 1024, 4 * 1024 * 1024);

static void BM_CudaMD5_Large(benchmark::State& state) {
    std::string message(state.range(0), 'd');
    for (auto _ : state) {
        auto result = hash_cuda(message);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
}
BENCHMARK(BM_CudaMD5_Large)->RangeMultiplier(8)->Range(2 * 1024 * 1024, 4 * 1024UL * 1024UL * 1024UL);

static void BM_CudaKernel(benchmark::State& state) {
    const std::vector<uint8_t> h_data_in(state.range(0),0xAB);
    std::vector<uint8_t> padded_message = preprocess(h_data_in.data(),h_data_in.size());
    uint64_t padded_size = padded_message.size();
    uint64_t num_chunks = padded_size / 64;

    uint8_t* d_padded_message;
    cudaMalloc(&d_padded_message, padded_size);
    cudaMemcpy(d_padded_message, padded_message.data(), padded_size, cudaMemcpyHostToDevice);
    uint32_t* d_states;
    cudaMalloc(&d_states, num_chunks * 4 * sizeof(uint32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        process_chunks_kernel<<<state.range(1), state.range(2)>>>(d_padded_message, state.range(0),d_states);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);
        state.SetIterationTime(elapsed_time_ms / 1000.0); // Set time per iteration in seconds
    }

    // Set bytes processed
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0) * sizeof(float));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_padded_message);
    cudaFree(d_states);
}
BENCHMARK(BM_CudaKernel)
    ->Range(1024, 1024 * 1024) // Data size
    ->Ranges({{1, 256}, {1, 1024}}) // Block and thread dimensions
    ->Unit(benchmark::kMicrosecond);