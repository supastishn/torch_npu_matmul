#ifndef QQ_FIXEDPOINT_H
#define QQ_FIXEDPOINT_H
#include "utils/execution.h"
#include <cstdint>
#include <cstring>
#include <mutex>
#include <cmath>
#include <limits>
#include <new>
#include <queue>
#include <future>
#include <vector>
#include <algorithm>
#include <cstdlib>
inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
inline float fast_pow2(int32_t shift) {
    union {
        int32_t i_val;
        float f_val;
    } u;
    u.i_val = (127 + shift) << 23;
    return u.f_val;
}
inline uint32_t calculate_scaled_fixed_value(uint32_t mantissa, int bit_shift) {
    uint32_t right_shift_amount = (bit_shift >= 0) ? bit_shift : 0;
    uint32_t left_shift_amount = (bit_shift < 0) ? -bit_shift : 0;
    uint32_t right_scaled = (bit_shift < 32) ? (mantissa >> right_shift_amount) : 0;
    uint32_t left_scaled = (bit_shift > -32) ? (mantissa << left_shift_amount) : 0;
    return (bit_shift >= 0) ? right_scaled : left_scaled;
}
template<typename OutType>
inline OutType process_single_stochastic_round(float input_value, uint32_t noise, int precision, uint32_t fractional_mask, bool& is_zero) {
    union { float f; uint32_t u; } converter;
    converter.f = input_value;
    uint32_t raw_bits = converter.u;
    uint32_t absolute_raw_bits = raw_bits & 0x7FFFFFFF;
    uint32_t exponent_bits = (absolute_raw_bits >> 23) & 0xFF;
    is_zero = (exponent_bits == 0);
    uint32_t random_bits = noise & fractional_mask;
    uint32_t mantissa_value = (1U << 23) | (absolute_raw_bits & 0x7FFFFF);
    int bit_shift = 150 - precision - exponent_bits;
    uint32_t scaled_fixed_value = calculate_scaled_fixed_value(mantissa_value, bit_shift);
    uint32_t fractional_bits_mask = scaled_fixed_value & fractional_mask;
    uint32_t integer_part_value = scaled_fixed_value >> precision;
    uint32_t rounded_magnitude = integer_part_value + ((fractional_bits_mask + random_bits) >> precision);
    int32_t sign_bit_mask = (int32_t)raw_bits >> 31;
    int32_t rounded_integer = (rounded_magnitude + sign_bit_mask) ^ sign_bit_mask;
    return (OutType)rounded_integer;
}
template<typename OutType>
inline void stochastic_round_array(const float* __restrict__ input_array, OutType* __restrict__ output_array, int array_size, int precision = 8) {
    uint32_t r0 = 2463534242U;
    uint32_t r1 = 4123546731U;
    uint32_t r2 = 1290384712U;
    uint32_t r3 = 3812403981U;
    uint32_t fractional_mask = (1U << precision) - 1;
    for (int i = 0; i < array_size - 3; i += 4) {
        uint32_t noise0 = xorshift32(r0);
        uint32_t noise1 = xorshift32(r1);
        uint32_t noise2 = xorshift32(r2);
        uint32_t noise3 = xorshift32(r3);
        uint32_t random_noises[4] = { noise0, noise1, noise2, noise3 };
        for (int j = 0; j < 4; ++j) {
            int flat_index = i + j;
            bool is_zero = false;
            OutType result = process_single_stochastic_round<OutType>(input_array[flat_index], random_noises[j], precision, fractional_mask, is_zero);
            output_array[flat_index] = is_zero ? (OutType)0 : result;
        }
    }
    int remainder_start = (array_size / 4) * 4;
    for (int i = remainder_start; i < array_size; ++i) {
        uint32_t noise = xorshift32(r0);
        bool is_zero = false;
        OutType result = process_single_stochastic_round<OutType>(input_array[i], noise, precision, fractional_mask, is_zero);
        output_array[i] = is_zero ? (OutType)0 : result;
    }
}
struct alignas(128) BlockHeader {
    size_t size;
    bool is_free;
    BlockHeader* next;
    BlockHeader* prev;
};
class Scratchpad {
private:
    inline static uint8_t* memory_buffer = nullptr;
    inline static size_t total_size = 32 * 1024 * 1024;
    inline static BlockHeader* head_block = nullptr;
    inline static std::mutex scratchpad_mutex;
    static void initialize_if_needed() {
        if (!memory_buffer) {
            load_rpcmem_once();
            if (rpcmem_loaded) {
                void* raw_mem = global_rpcmem_alloc(25, 0, total_size);
                if (raw_mem) {
                    memory_buffer = (uint8_t*)raw_mem;
                }
            }
            if (!memory_buffer) {
                void* raw_mem = nullptr;
                int res = posix_memalign(&raw_mem, 128, total_size);
                if (res == 0) {
                    memory_buffer = (uint8_t*)raw_mem;
                }
            }
            if (memory_buffer) {
                head_block = (BlockHeader*)memory_buffer;
                head_block->size = total_size - sizeof(BlockHeader);
                head_block->is_free = true;
                head_block->next = nullptr;
                head_block->prev = nullptr;
            }
        }
    }
public:
    static void set_size(size_t new_size) {
        std::lock_guard<std::mutex> lock(scratchpad_mutex);
        if (memory_buffer) {
            load_rpcmem_once();
            if (rpcmem_loaded) {
                global_rpcmem_free(memory_buffer);
            } else {
                ::free(memory_buffer);
            }
            memory_buffer = nullptr;
            head_block = nullptr;
        }
        total_size = new_size;
    }
    static int get_fd_and_offset(void* ptr, uint32_t& offset) {
        load_rpcmem_once();
        if (rpcmem_loaded && memory_buffer && ptr >= memory_buffer && ptr < memory_buffer + total_size) {
            offset = (uint32_t)((uint8_t*)ptr - memory_buffer);
            return global_rpcmem_to_fd(memory_buffer);
        }
        offset = 0;
        return -1;
    }
    static void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(scratchpad_mutex);
        initialize_if_needed();
        size = (size + 127) & ~127;
        BlockHeader* current_block = head_block;
        while (current_block) {
            if (current_block->is_free && current_block->size >= size) {
                if (current_block->size >= size + sizeof(BlockHeader) + 128) {
                    BlockHeader* next_block = (BlockHeader*)((uint8_t*)current_block + sizeof(BlockHeader) + size);
                    next_block->size = current_block->size - size - sizeof(BlockHeader);
                    next_block->is_free = true;
                    next_block->next = current_block->next;
                    next_block->prev = current_block;
                    if (current_block->next) current_block->next->prev = next_block;
                    current_block->next = next_block;
                    current_block->size = size;
                }
                current_block->is_free = false;
                return (void*)((uint8_t*)current_block + sizeof(BlockHeader));
            }
            current_block = current_block->next;
        }
        void* raw_mem = nullptr;
        posix_memalign(&raw_mem, 128, size);
        return raw_mem;
    }
    static void free(void* allocated_pointer) {
        if (!allocated_pointer) return;
        std::lock_guard<std::mutex> lock(scratchpad_mutex);
        if (memory_buffer && (uint8_t*)allocated_pointer >= memory_buffer && (uint8_t*)allocated_pointer < memory_buffer + total_size) {
            BlockHeader* current_block = (BlockHeader*)((uint8_t*)allocated_pointer - sizeof(BlockHeader));
            current_block->is_free = true;
            if (current_block->next && current_block->next->is_free) {
                current_block->size += sizeof(BlockHeader) + current_block->next->size;
                current_block->next = current_block->next->next;
                if (current_block->next) current_block->next->prev = current_block;
            }
            if (current_block->prev && current_block->prev->is_free) {
                current_block->prev->size += sizeof(BlockHeader) + current_block->size;
                current_block->prev->next = current_block->next;
                if (current_block->next) current_block->next->prev = current_block->prev;
            }
        } else {
            ::free(allocated_pointer);
        }
    }
};
struct ScratchpadBuffer {
    float* pointer;
    ScratchpadBuffer(size_t size) {
        pointer = (float*)Scratchpad::alloc(size * sizeof(float));
    }
    ~ScratchpadBuffer() {
        Scratchpad::free(pointer);
    }
};
template<typename T>
struct FixedPointBlock {
    T* mantissa;
    int32_t* exponents;
    float* means;
    float* std_devs;
    int* precisions;
    int rows;
    int cols;
    int block_size;
    bool block_columns;
    int num_blocks;
    float mean;
    float std_dev;
    FixedPointBlock(int num_rows, int num_columns, int target_block_size, bool use_block_columns, size_t min_bytes = 0) {
        rows = num_rows;
        cols = num_columns;
        block_size = target_block_size;
        block_columns = use_block_columns;
        size_t alloc_bytes = rows * cols * sizeof(T);
        if (min_bytes > alloc_bytes) {
            alloc_bytes = min_bytes;
        }
        mantissa = (T*)Scratchpad::alloc(alloc_bytes);
        if (block_columns) {
            num_blocks = (cols + block_size - 1) / block_size;
            exponents = (int32_t*)Scratchpad::alloc(rows * num_blocks * sizeof(int32_t));
        } else {
            num_blocks = (rows + block_size - 1) / block_size;
            exponents = (int32_t*)Scratchpad::alloc(num_blocks * cols * sizeof(int32_t));
        }
        int total_blocks = block_columns ? (rows * num_blocks) : (num_blocks * cols);
        means = (float*)Scratchpad::alloc(total_blocks * sizeof(float));
        std_devs = (float*)Scratchpad::alloc(total_blocks * sizeof(float));
        precisions = (int*)Scratchpad::alloc(total_blocks * sizeof(int));
    }
    ~FixedPointBlock() {
        Scratchpad::free(mantissa);
        Scratchpad::free(exponents);
        Scratchpad::free(means);
        Scratchpad::free(std_devs);
        Scratchpad::free(precisions);
    }
    void fit_exponent(const float* values) {
        float global_sum = 0.0f;
        float global_sum_squares = 0.0f;
        int global_n = 0;
        if (block_columns) {
            int num_blocks_count = (cols + block_size - 1) / block_size;
            #pragma omp parallel for reduction(+:global_sum, global_sum_squares, global_n) if(rows > 1)
            for (int row_index = 0; row_index < rows; ++row_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    uint32_t max_biased = 0;
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    float block_sum = 0.0f;
                    float block_sum_squares = 0.0f;
                    int block_n = 0;
                    for (int col_index = start_col; col_index < end_col; ++col_index) {
                        float val = values[row_index * cols + start_col + (col_index - start_col)];
                        global_sum += val;
                        global_sum_squares += val * val;
                        ++global_n;
                        block_sum += val;
                        block_sum_squares += val * val;
                        ++block_n;
                        uint32_t absolute_raw_bits;
                        std::memcpy(&absolute_raw_bits, &val, sizeof(uint32_t));
                        absolute_raw_bits &= 0x7FFFFFFF;
                        uint32_t biased = (absolute_raw_bits >> 23) & 0xFF;
                        if (biased > max_biased) {
                            max_biased = biased;
                        }
                    }
                    int block_flat_idx = row_index * num_blocks_count + block_index;
                    exponents[block_flat_idx] = (max_biased > 0) ? ((int32_t)max_biased - 127 + 1) : -127;
                    if (block_n > 0) {
                        means[block_flat_idx] = block_sum / block_n;
                        float variance = (block_sum_squares / block_n) - (means[block_flat_idx] * means[block_flat_idx]);
                        std_devs[block_flat_idx] = std::sqrt(std::max(0.0f, variance));
                    } else {
                        means[block_flat_idx] = 0.0f;
                        std_devs[block_flat_idx] = 0.0f;
                    }
                }
            }
        } else {
            int num_blocks_count = (rows + block_size - 1) / block_size;
            #pragma omp parallel for reduction(+:global_sum, global_sum_squares, global_n) if(cols > 1)
            for (int col_index = 0; col_index < cols; ++col_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    uint32_t max_biased = 0;
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    float block_sum = 0.0f;
                    float block_sum_squares = 0.0f;
                    int block_n = 0;
                    for (int row_index = start_row; row_index < end_row; ++row_index) {
                        float val = values[row_index * cols + col_index];
                        global_sum += val;
                        global_sum_squares += val * val;
                        ++global_n;
                        block_sum += val;
                        block_sum_squares += val * val;
                        ++block_n;
                        uint32_t absolute_raw_bits;
                        std::memcpy(&absolute_raw_bits, &val, sizeof(uint32_t));
                        absolute_raw_bits &= 0x7FFFFFFF;
                        uint32_t biased = (absolute_raw_bits >> 23) & 0xFF;
                        if (biased > max_biased) {
                            max_biased = biased;
                        }
                    }
                    int block_flat_idx = block_index * cols + col_index;
                    exponents[block_flat_idx] = (max_biased > 0) ? ((int32_t)max_biased - 127 + 1) : -127;
                    if (block_n > 0) {
                        means[block_flat_idx] = block_sum / block_n;
                        float variance = (block_sum_squares / block_n) - (means[block_flat_idx] * means[block_flat_idx]);
                        std_devs[block_flat_idx] = std::sqrt(std::max(0.0f, variance));
                    } else {
                        means[block_flat_idx] = 0.0f;
                        std_devs[block_flat_idx] = 0.0f;
                    }
                }
            }
        }
        if (global_n > 0) {
            mean = global_sum / global_n;
            float global_variance = (global_sum_squares / global_n) - (mean * mean);
            std_dev = std::sqrt(std::max(0.0f, global_variance));
        } else {
            mean = 0.0f;
            std_dev = 0.0f;
        }
    }
    void floats_to_mantissa(const float* floats_array, int fixed_exponent, int force_precision = 0, float z_sigma = 3.0f, int other_exponent = 8, float other_std_dev = 0.5f, float other_mean = 0.0f) { 
        if (block_columns) {
            int num_blocks_count = (cols + block_size - 1) / block_size;
            #pragma omp parallel for if(rows > 1)
            for (int row_index = 0; row_index < rows; ++row_index) {
                float* temporary_buffer = new float[cols];
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int block_flat_idx = row_index * num_blocks_count + block_index;
                    int32_t exponent_val = (fixed_exponent != 0) ? fixed_exponent : exponents[block_flat_idx];
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    int count = end_col - start_col;
                    int bits_of_precision;
                    if (force_precision > 0) {
                        bits_of_precision = force_precision;
                    } else {
                        bits_of_precision = sizeof(T) * 8 - 1;
                        bool does_not_fit = true;
                        float b_std_dev_norm = std_devs[block_flat_idx] / fast_pow2(exponent_val);
                        while (does_not_fit && bits_of_precision > 1) {
                            float std_dev_output = std::sqrt((float)count) * 
                                                   (other_std_dev * (fast_pow2(other_exponent - 1) - 1.0f)) * 
                                                   (b_std_dev_norm * (fast_pow2(bits_of_precision)));
                            does_not_fit = (z_sigma * std_dev_output > 32767.0f);
                            if (does_not_fit) {
                                --bits_of_precision;
                            }
                        }
                    }
                    precisions[block_flat_idx] = bits_of_precision;
                    float exponent_scale = fast_pow2(-exponent_val);
                    const float* __restrict__ source_ptr = floats_array + row_index * cols + start_col;
                    float* __restrict__ dest_ptr = temporary_buffer;
                    #pragma clang loop vectorize(enable)
                    for (int col_index = 0; col_index < count; ++col_index) {
                        dest_ptr[col_index] = source_ptr[col_index] * exponent_scale;
                    }
                    stochastic_round_array(temporary_buffer, &mantissa[row_index * cols + start_col], count, bits_of_precision);
                    float integer_mean_float = means[block_flat_idx] * fast_pow2(-exponent_val + bits_of_precision);
                    T integer_mean = (T)std::round(integer_mean_float);
                    #pragma clang loop vectorize(enable)
                    for (int col_index = 0; col_index < count; ++col_index) {
                        mantissa[row_index * cols + start_col + col_index] = mantissa[row_index * cols + start_col + col_index] - integer_mean;
                    }
                }
                delete[] temporary_buffer;
            }
        } else {
            int num_blocks_count = (rows + block_size - 1) / block_size;
            #pragma omp parallel for if(cols > 1)
            for (int col_index = 0; col_index < cols; ++col_index) {
                float* temporary_buffer = new float[rows];
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int block_flat_idx = block_index * cols + col_index;
                    int32_t exponent_val = (fixed_exponent != 0) ? fixed_exponent : exponents[block_flat_idx];
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    int count = end_row - start_row;
                    int bits_of_precision;
                    if (force_precision > 0) {
                        bits_of_precision = force_precision;
                    } else {
                        bits_of_precision = sizeof(T) * 8 - 1;
                        bool does_not_fit = true;
                        float b_std_dev_norm = std_devs[block_flat_idx] / fast_pow2(exponent_val);
                        while (does_not_fit && bits_of_precision > 1) {
                            float std_dev_output = std::sqrt((float)count) * 
                                                   (other_std_dev * (fast_pow2(other_exponent - 1) - 1.0f)) * 
                                                   (b_std_dev_norm * (fast_pow2(bits_of_precision)));
                            does_not_fit = (z_sigma * std_dev_output > 32767.0f);
                            if (does_not_fit) {
                                --bits_of_precision;
                            }
                        }
                    }
                    precisions[block_flat_idx] = bits_of_precision;
                    float exponent_scale = fast_pow2(-exponent_val);
                    const float* __restrict__ source_ptr = floats_array + start_row * cols + col_index;
                    float* __restrict__ dest_ptr = temporary_buffer;
                    for (int row_index = 0; row_index < count; ++row_index) {
                        dest_ptr[row_index] = source_ptr[row_index * cols] * exponent_scale;
                    }
                    T* column_mantissa = new T[count];
                    stochastic_round_array(temporary_buffer, column_mantissa, count, bits_of_precision);
                    float integer_mean_float = means[block_flat_idx] * fast_pow2(-exponent_val + bits_of_precision);
                    T integer_mean = (T)std::round(integer_mean_float);
                    #pragma clang loop vectorize(enable)
                    for (int row_index = 0; row_index < count; ++row_index) {
                        mantissa[(start_row + row_index) * cols + col_index] = column_mantissa[row_index] - integer_mean;
                    }
                    delete[] column_mantissa;
                }
                delete[] temporary_buffer;
            }
        }
    }
    void mantissa_to_floats(float* output_array) {
        if (block_columns) {
            int num_blocks_count = (cols + block_size - 1) / block_size;
            #pragma omp parallel for if(rows > 1)
            for (int row_index = 0; row_index < rows; ++row_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int block_flat_idx = row_index * num_blocks_count + block_index;
                    int32_t exponent_val = exponents[block_flat_idx];
                    float exponent_scale = fast_pow2(exponent_val - precisions[block_flat_idx]);
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    #pragma clang loop vectorize(enable)
                    for (int col_index = start_col; col_index < end_col; ++col_index) {
                        output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale + means[block_flat_idx];
                    }
                }
            }
        } else {
            int num_blocks_count = (rows + block_size - 1) / block_size;
            #pragma omp parallel for if(cols > 1)
            for (int col_index = 0; col_index < cols; ++col_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int block_flat_idx = block_index * cols + col_index;
                    int32_t exponent_val = exponents[block_flat_idx];
                    float exponent_scale = fast_pow2(exponent_val - precisions[block_flat_idx]);
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    #pragma clang loop vectorize(enable)
                    for (int row_index = start_row; row_index < end_row; ++row_index) {
                        output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale + means[block_flat_idx];
                    }
                }
            }
        }
    }
    void mantissa_to_floats_product(float* output_array, int other_exponent = 8) {
        if (block_columns) {
            int num_blocks_count = (cols + block_size - 1) / block_size;
            #pragma omp parallel for if(rows > 1)
            for (int row_index = 0; row_index < rows; ++row_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int block_flat_idx = row_index * num_blocks_count + block_index;
                    int32_t exponent_val = exponents[block_flat_idx];
                    float exponent_scale = fast_pow2(exponent_val - precisions[block_flat_idx] - other_exponent);
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    float mean_part = means[block_flat_idx] * fast_pow2(-other_exponent);
                    #pragma clang loop vectorize(enable)
                    for (int col_index = start_col; col_index < end_col; ++col_index) {
                        output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale + mean_part;
                    }
                }
            }
        } else {
            int num_blocks_count = (rows + block_size - 1) / block_size;
            #pragma omp parallel for if(cols > 1)
            for (int col_index = 0; col_index < cols; ++col_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int block_flat_idx = block_index * cols + col_index;
                    int32_t exponent_val = exponents[block_flat_idx];
                    float exponent_scale = fast_pow2(exponent_val - precisions[block_flat_idx] - other_exponent);
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    float mean_part = means[block_flat_idx] * fast_pow2(-other_exponent);
                    #pragma clang loop vectorize(enable)
                    for (int row_index = start_row; row_index < end_row; ++row_index) {
                        output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale + mean_part;
                    }
                }
            }
        }
    }
};
class PrefetchQueue {
private:
    std::queue<FixedPointBlock<int16_t>*> queue;
    std::mutex mutex;
public:
    void push(FixedPointBlock<int16_t>* block) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(block);
    }
    FixedPointBlock<int16_t>* pop() {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) return nullptr;
        FixedPointBlock<int16_t>* block = queue.front();
        queue.pop();
        return block;
    }
    void clear_queue() {
        std::lock_guard<std::mutex> lock(mutex);
        while (!queue.empty()) {
            delete queue.front();
            queue.pop();
        }
    }
};
inline PrefetchQueue prefetch_queue;
inline std::vector<std::future<void>> active_prefetch_futures;
inline void prefetch_weights(const std::vector<const float*>& weight_ptrs, const std::vector<int>& next_sizes) {
    for (size_t i = 0; i < weight_ptrs.size() && i < next_sizes.size(); ++i) {
        const float* ptr = weight_ptrs[i];
        int size = next_sizes[i];
        active_prefetch_futures.push_back(std::async(std::launch::async, [ptr, size]() {
            FixedPointBlock<int16_t>* block = new FixedPointBlock<int16_t>(size, size, size, false);
            block->fit_exponent(ptr);
            block->floats_to_mantissa(ptr, 0, 8);
            prefetch_queue.push(block);
        }));
    }
}
inline void prefetch_weights(const std::vector<const float*>& weight_ptrs, const std::vector<int>& rows, const std::vector<int>& cols) {
    for (size_t i = 0; i < weight_ptrs.size() && i < rows.size() && i < cols.size(); ++i) {
        const float* ptr = weight_ptrs[i];
        int r = rows[i];
        int c = cols[i];
        active_prefetch_futures.push_back(std::async(std::launch::async, [ptr, r, c]() {
            FixedPointBlock<int16_t>* block = new FixedPointBlock<int16_t>(r, c, r, false);
            block->fit_exponent(ptr);
            block->floats_to_mantissa(ptr, 0, 8);
            prefetch_queue.push(block);
        }));
    }
}
inline void clear_queue() {
    prefetch_queue.clear_queue();
    active_prefetch_futures.clear();
}
#endif