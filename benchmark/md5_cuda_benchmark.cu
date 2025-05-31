#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "md5_cuda.cuh"

static void BM_CudaMD5(benchmark::State& state) {
    std::string message(state.range(0), 'd');
    for (auto _ : state) {
        auto result = hash_cuda(message);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0));
}
BENCHMARK(BM_CudaMD5)->Range(1024, 2*1024UL * 1024UL * 1024UL);


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
    const int threadsPerBlock = state.range(1);
    int numBlocks = (num_chunks + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start, 0);
        process_chunks_kernel<<<numBlocks, threadsPerBlock>>>(d_padded_message, state.range(0),d_states);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        benchmark::DoNotOptimize(d_states);
        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);
        state.SetIterationTime(elapsed_time_ms / 1000.0); // Set time per iteration in seconds
    }

    // // Set bytes processed
    state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0) * sizeof(float));
    // state.SetLabel(std::to_string(state.range(0)));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_padded_message);
    cudaFree(d_states);
}
BENCHMARK(BM_CudaKernel)
    // ->MinWarmUpTime(10)
    ->ArgsProduct({
        benchmark::CreateRange(1024, 1024UL * 1024UL * 1024UL, 2), // Data size
        {16, 32, 64, 128, 256, 512, 1024} // Block and thread dimensions
    })
    ->Unit(benchmark::kMicrosecond);