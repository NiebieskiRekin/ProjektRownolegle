#include "md5_cuda.cuh"

// Constants
__device__ const uint32_t s_dev[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

__device__ const uint32_t K_dev[64] = {
    3614090360U, 3905402710U, 606105819U,  3250441966U, 4118548399U, 1200080426U,
    2821735955U, 4249261313U, 1770035416U, 2336552879U, 4294925233U, 2304563134U,
    1804603682U, 4254626195U, 2792965006U, 1236535329U, 4129170786U, 3225465664U,
    643717713U,  3921069994U, 3593408605U, 38016083U,   3634488961U, 3889429448U,
    568446438U,  3275163606U, 4107603335U, 1163531501U, 2850285829U, 4243563512U,
    1735328473U, 2368359562U, 4294588738U, 2272392833U, 1839030562U, 4259657740U,
    2763975236U, 1272893353U, 4139469664U, 3200236656U, 681279174U,  3936430074U,
    3572445317U, 76029189U,   3654602809U, 3873151461U, 530742520U,  3299628645U,
    4096336452U, 1126891415U, 2878612391U, 4237533241U, 1700485571U, 2399980690U,
    4293915773U, 2240044497U, 1873313359U, 4264355552U, 2734768916U, 1309151649U,
    4149444226U, 3174756917U, 718787259U,  3951481745U
};
__device__ const uint32_t initial_128_bit_state_dev[4] = {0x67452301U, 0xefcdab89U, 0x98badcfeU, 0x10325476U};


// Device function for left rotation
__device__ inline uint32_t leftRotate32bits_device(uint32_t n, uint32_t rotate) {
    return (n << rotate) | (n >> (32 - rotate));
}

// CUDA kernel to process multiple chunks in parallel
__global__ void process_chunks_kernel(
    const uint8_t* d_padded_message,
    uint64_t num_chunks,
    uint32_t* d_states
) {
    uint64_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (chunk_id < num_chunks) {
        uint64_t chunk_start = chunk_id * 64;
        uint32_t blocks[16]{};
        uint32_t a0 = initial_128_bit_state_dev[0];
        uint32_t b0 = initial_128_bit_state_dev[1];
        uint32_t c0 = initial_128_bit_state_dev[2];
        uint32_t d0 = initial_128_bit_state_dev[3];
        uint32_t A = a0;
        uint32_t B = b0;
        uint32_t C = c0;
        uint32_t D = d0;

        // First, build the 16 32-bits blocks from the chunk
        for (uint8_t bid = 0; bid < 16; bid++) {
            blocks[bid] = 0;
            for (uint8_t cid = 0; cid < 4; cid++) {
                blocks[bid] = (blocks[bid] << 8) + d_padded_message[chunk_start + bid * 4 + cid];
            }
        }

        // Main "hashing" loop
        for (uint8_t i = 0; i < 64; i++) {
            uint32_t F = 0, g = 0;
            if (i < 16) {
                F = (B & C) | ((~B) & D);
                g = i;
            } else if (i < 32) {
                F = (D & B) | ((~D) & C);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                F = B ^ C ^ D;
                g = (3 * i + 5) % 16;
            } else {
                F = C ^ (B | (~D));
                g = (7 * i) % 16;
            }

            // Update the accumulators
            F += A + K_dev[i] + blocks[g];

            A = D;
            D = C;
            C = B;
            B += leftRotate32bits_device(F, s_dev[i]);
        }
        // Store the intermediate state for this chunk
        d_states[chunk_id * 4 + 0] = A;
        d_states[chunk_id * 4 + 1] = B;
        d_states[chunk_id * 4 + 2] = C;
        d_states[chunk_id * 4 + 3] = D;
    }
}


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