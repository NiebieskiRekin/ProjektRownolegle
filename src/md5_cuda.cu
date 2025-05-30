#include "md5_cuda.cuh"

std::array<uint8_t, 16> hash_cuda(const void* input_bs, uint64_t input_size) {
    const uint8_t* input = static_cast<const uint8_t*>(input_bs);

    // Preprocess the input on the host
    std::vector<uint8_t> padded_message = preprocess(input, input_size);
    uint64_t padded_size = padded_message.size();
    uint64_t num_chunks = padded_size / 64;

    // Allocate device memory for the padded message
    uint8_t* d_padded_message;
    cudaMalloc((void**)&d_padded_message, padded_size);
    cudaMemcpy(d_padded_message, padded_message.data(), padded_size, cudaMemcpyHostToDevice);

    // Allocate device memory for the intermediate states
    uint32_t* d_states;
    cudaMalloc((void**)&d_states, num_chunks * 4 * sizeof(uint32_t));

    // Configure the grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (num_chunks + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    process_chunks_kernel<<<numBlocks, threadsPerBlock>>>(
        d_padded_message,
        num_chunks,
        d_states
    );

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_padded_message);
        cudaFree(d_states);
        return std::array<uint8_t, 16>();
    }

    // Copy the intermediate states back to the host
    std::vector<uint32_t> h_states(num_chunks * 4);
    cudaMemcpy(h_states.data(), d_states, num_chunks * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Perform the final sequential accumulation on the host
    std::array<uint32_t, 4> final_state = initial_128_bit_state;
    for (uint64_t i = 0; i < num_chunks; ++i) {
        final_state[0] += h_states[i * 4 + 0];
        final_state[1] += h_states[i * 4 + 1];
        final_state[2] += h_states[i * 4 + 2];
        final_state[3] += h_states[i * 4 + 3];
    }

    // Build the signature on the host
    auto signature = build_signature(final_state[0], final_state[1], final_state[2], final_state[3]);

    // Free device memory
    cudaFree(d_padded_message);
    cudaFree(d_states);

    return signature;
}

std::array<uint8_t, 16> hash_cuda(const std::string& message) {
    return hash_cuda(message.data(), message.size());
}