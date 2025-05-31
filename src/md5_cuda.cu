#include "md5_cuda.cuh"

// Constants
__device__ const uint32_t s_dev[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

__device__ const uint32_t K_dev[64] = {
    3614090360, 3905402710, 606105819, 3250441966, 4118548399, 1200080426,
    2821735955, 4249261313, 1770035416, 2336552879, 4294925233, 2304563134,
    1804603682, 4254626195, 2792965006, 1236535329, 4129170786, 3225465664,
    643717713, 3921069994, 3593408605, 38016083, 3634488961, 3889429448,
    568446438, 3275163606, 4107603335, 1163531501, 2850285829, 4243563512,
    1735328473, 2368359562, 4294588738, 2272392833, 1839030562, 4259657740,
    2763975236, 1272893353, 4139469664, 3200236656, 681279174, 3936430074,
    3572445317, 76029189, 3654602809, 3873151461, 530742520, 3299628645,
    4096336452, 1126891415, 2878612391, 4237533241, 1700485571, 2399980690,
    4293915773, 2240044497, 1873313359, 4264355552, 2734768916, 1309151649,
    4149444226, 3174756917, 718787259, 3951481745
};
__device__ const uint32_t initial_128_bit_state_dev[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};


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
        uint32_t A = initial_128_bit_state_dev[0];
        uint32_t B = initial_128_bit_state_dev[1];
        uint32_t C = initial_128_bit_state_dev[2];
        uint32_t D = initial_128_bit_state_dev[3];

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

void combineStates(std::vector<uint32_t>& chunk_states, uint64_t i, uint64_t next) {
    chunk_states[i+0] = chunk_states[i+0] + chunk_states[i+next+0];
    chunk_states[i+1] = chunk_states[i+1] + chunk_states[i+next+1];
    chunk_states[i+2] = chunk_states[i+2] + chunk_states[i+next+2];
    chunk_states[i+3] = chunk_states[i+3] + chunk_states[i+next+3];
}

void merkleTreeReduce(std::vector<uint32_t>& chunk_states){
    // Merkle tree-like reduction of chunk states
    uint64_t num_chunks = chunk_states.size() /4;
    uint64_t num_levels = std::ceil(std::log2(num_chunks));
    uint64_t next = 1;
    uint64_t num_nodes_at_level = num_chunks;
    for (uint64_t level = 1; level <= num_levels; ++level) {
        uint64_t skip = next<<1;
        for (uint64_t i = 0; i+next < num_nodes_at_level; i += skip) {
            combineStates(chunk_states, i, next);
        }   

        num_nodes_at_level = (num_nodes_at_level>>1) + (num_nodes_at_level%2);
        next = skip;
    }
}


std::array<uint8_t, 16> hash_cuda(const void* input_bs, uint64_t input_size, int threadsPerBlock) {
    const uint8_t* input = static_cast<const uint8_t*>(input_bs);

    // Preprocess the input on the host
    std::vector<uint8_t> padded_message = preprocess(input, input_size);
    uint64_t padded_size = padded_message.size();
    uint64_t num_chunks = padded_size / 64;

    // Allocate device memory for the padded message
    uint8_t* d_padded_message;
    cudaMalloc(&d_padded_message, padded_size);
    cudaMemcpy(d_padded_message, padded_message.data(), padded_size, cudaMemcpyHostToDevice);

    // Allocate device memory for the intermediate states
    uint32_t* d_states;
    cudaMalloc(&d_states, num_chunks * 4 * sizeof(uint32_t));

    // Configure the grid and block dimensions
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
    merkleTreeReduce(h_states);
    auto signature = build_signature(h_states[0], h_states[1], h_states[2], h_states[3]);

    // Free device memory
    cudaFree(d_padded_message);
    cudaFree(d_states);

    return signature;
}



std::array<uint8_t, 16> hash_cuda(const void* input_bs, uint64_t input_size) {
    return hash_cuda(input_bs,input_size,1024);
}

std::array<uint8_t, 16> hash_cuda(const std::string& message) {
    return hash_cuda(message.data(), message.size());
}