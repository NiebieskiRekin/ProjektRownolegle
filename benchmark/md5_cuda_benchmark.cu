// #include <benchmark/benchmark.h>
// #include <iostream>
// #include <vector>
// #include <cuda_runtime.h>

// static void BM_CudaKernel(benchmark::State& state) {
//     // Allocate device memory (if needed)
//     float *d_data;
//     size_t size = state.range(0) * sizeof(float);
//     cudaMalloc(&d_data, size);

//     // Initialize host data (if needed)
//     std::vector<float> h_data(state.range(0));
//     // ... initialize h_data ...
//     cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     for (auto _ : state) {
//         cudaEventRecord(start, 0);
//         your_kernel<<<state.range(1), state.range(2)>>>(d_data, state.range(0));
//         cudaEventRecord(stop, 0);
//         cudaEventSynchronize(stop);
//         float elapsed_time_ms;
//         cudaEventElapsedTime(&elapsed_time_ms, start, stop);
//         state.SetIterationTime(elapsed_time_ms / 1000.0); // Set time per iteration in seconds
//     }

//     // Set bytes processed (if applicable)
//     state.SetBytesProcessed(int64_t(state.iterations()) * state.range(0) * sizeof(float));

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     cudaFree(d_data);
// }
// BENCHMARK(BM_CudaKernel)
//     ->Range(1024, 1024 * 1024) // Data size
//     ->Ranges({{1, 256}, {1, 1024}}) // Block and thread dimensions
//     ->Unit(benchmark::kMicrosecond);