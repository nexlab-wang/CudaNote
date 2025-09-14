#include "cuda_runtime.h"
#include "common/buffer.h"
#include <iostream>
#include <vector>

#define ITERATIONS 10

void soft_max_host(const float *input, float *output, size_t size)
{
    float max_val = -INFINITY;
    for (size_t i = 0; i < size; ++i)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    for (size_t i = 0; i < size; ++i)
    {
        output[i] /= sum;
    }
}

__global__ void soft_max_naive_kernel(const float *input, float *output, size_t size)
{
    int tid = threadIdx.x;

    // Only thread 0 performs all computations
    if (tid == 0)
    {
        float global_max = -FLT_MAX;
        for (int i = 0; i < size; i++)
        {
            global_max = fmaxf(global_max, input[i]);
        }

        float global_sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            global_sum += expf(input[i] - global_max);
        }

        for (int i = 0; i < size; i++)
        {
            output[i] = expf(input[i] - global_max) / global_sum;
        }
    }
}

__device__ float warp_reduce_max(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Optimized softmax kernel using warp-level primitives
 */
__global__ void soft_max_reduce_kernel(const float *input, float *output, size_t size)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ float shared_mem[];
    float *warp_max = shared_mem;
    float *warp_sum = shared_mem + (blockDim.x / 32);

    float thread_max = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x)
    {
        if (input[i] > thread_max)
            thread_max = input[i];
    }

    float warp_max_val = warp_reduce_max(thread_max);
    if (lane_id == 0)
        warp_max[warp_id] = warp_max_val;
    __syncthreads();

    float global_max = -INFINITY;
    if (tid < (blockDim.x / 32))
        global_max = warp_max[tid];
    global_max = warp_reduce_max(global_max);

    if (tid == 0)
        warp_max[0] = global_max;
    __syncthreads();
    global_max = warp_max[0];

    float thread_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        thread_sum += expf(input[i] - global_max);
    }

    float warp_sum_val = warp_reduce_sum(thread_sum);
    if (lane_id == 0)
        warp_sum[warp_id] = warp_sum_val;
    __syncthreads();

    float global_sum = 0.0f;
    if (tid < (blockDim.x / 32))
        global_sum = warp_sum[tid];
    global_sum = warp_reduce_sum(global_sum);

    if (tid == 0)
        warp_sum[0] = global_sum;
    __syncthreads();
    global_sum = warp_sum[0];

    for (int i = tid; i < size; i += blockDim.x)
    {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

__global__ void soft_max_vectorized_kernel(const float *input, float *output, size_t size)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ float shared_mem[];
    float *warp_max = shared_mem;
    float *warp_sum = shared_mem + (blockDim.x / 32);

    // 使用float向量化加载优化
    float thread_max = -INFINITY;
    const int vector_size = 4;
    const int num_vectors = (size + vector_size - 1) / vector_size;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);

    for (int i = tid; i < num_vectors; i += blockDim.x)
    {
        float4 val4;
        int base_idx = i * vector_size;

        if (base_idx + 3 < size)
        {
            val4 = input4[i];
        }
        else
        {
            // 处理边界情况
            val4.x = (base_idx < size) ? input[base_idx] : -INFINITY;
            val4.y = (base_idx + 1 < size) ? input[base_idx + 1] : -INFINITY;
            val4.z = (base_idx + 2 < size) ? input[base_idx + 2] : -INFINITY;
            val4.w = (base_idx + 3 < size) ? input[base_idx + 3] : -INFINITY;
        }

        thread_max = max(thread_max, val4.x);
        thread_max = max(thread_max, val4.y);
        thread_max = max(thread_max, val4.z);
        thread_max = max(thread_max, val4.w);
    }

    float warp_max_val = warp_reduce_max(thread_max);
    if (lane_id == 0)
    {
        warp_max[warp_id] = warp_max_val;
    }
    __syncthreads();

    float global_max = -INFINITY;
    if (tid < (blockDim.x / 32))
    {
        global_max = warp_max[tid];
    }
    global_max = warp_reduce_max(global_max);

    if (tid == 0)
    {
        warp_max[0] = global_max;
    }
    __syncthreads();
    global_max = warp_max[0];

    // 使用float4向量化加载指数和计算
    float thread_sum = 0.0f;

    for (int i = tid; i < num_vectors; i += blockDim.x)
    {
        float4 val4;
        int base_idx = i * vector_size;

        if (base_idx + 3 < size)
        {
            val4 = input4[i];
        }
        else
        {
            val4.x = (base_idx < size) ? input[base_idx] : 0.0f;
            val4.y = (base_idx + 1 < size) ? input[base_idx + 1] : 0.0f;
            val4.z = (base_idx + 2 < size) ? input[base_idx + 2] : 0.0f;
            val4.w = (base_idx + 3 < size) ? input[base_idx + 3] : 0.0f;
        }

        // 计算指数和
        thread_sum += expf(val4.x - global_max);
        thread_sum += expf(val4.y - global_max);
        thread_sum += expf(val4.z - global_max);
        thread_sum += expf(val4.w - global_max);
    }

    float warp_sum_val = warp_reduce_sum(thread_sum);
    if (lane_id == 0)
    {
        warp_sum[warp_id] = warp_sum_val;
    }
    __syncthreads();

    float global_sum = 0.0f;
    if (tid < (blockDim.x / 32))
    {
        global_sum = warp_sum[tid];
    }
    global_sum = warp_reduce_sum(global_sum);

    if (tid == 0)
    {
        warp_sum[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_sum[0];

    // 使用float4向量化存储优化输出
    float4 *output4 = reinterpret_cast<float4 *>(output);
    float inv_global_sum = 1.0f / global_sum;

    for (int i = tid; i < num_vectors; i += blockDim.x)
    {
        float4 val4;
        float4 result4;
        int base_idx = i * vector_size;

        if (base_idx + 3 < size)
        {
            val4 = input4[i];
        }
        else
        {
            val4.x = (base_idx < size) ? input[base_idx] : 0.0f;
            val4.y = (base_idx + 1 < size) ? input[base_idx + 1] : 0.0f;
            val4.z = (base_idx + 2 < size) ? input[base_idx + 2] : 0.0f;
            val4.w = (base_idx + 3 < size) ? input[base_idx + 3] : 0.0f;
        }

        result4.x = expf(val4.x - global_max) * inv_global_sum;
        result4.y = expf(val4.y - global_max) * inv_global_sum;
        result4.z = expf(val4.z - global_max) * inv_global_sum;
        result4.w = expf(val4.w - global_max) * inv_global_sum;

        if (base_idx + 3 < size)
        {
            output4[i] = result4;
        }
        else
        {
            if (base_idx < size)
                output[base_idx] = result4.x;
            if (base_idx + 1 < size)
                output[base_idx + 1] = result4.y;
            if (base_idx + 2 < size)
                output[base_idx + 2] = result4.z;
            if (base_idx + 3 < size)
                output[base_idx + 3] = result4.w;
        }
    }
}

void soft_max(const std::string &kernel_name, void (*kernel)(const float *, float *, size_t),
              const float *input, float *output, const size_t &size)
{
    dim3 block(256);
    dim3 grid;

    grid = dim3((size + block.x - 1) / block.x);

    size_t shared_mem_size = 0;
    if (kernel_name == "soft_max_reduce_kernel")
    {
        size_t warps_per_block = (block.x + 31) / 32;
        shared_mem_size = 2 * warps_per_block * sizeof(float); // max+sum
    }
    else if (kernel_name == "soft_max_vectorized_kernel")
    {
        size_t warps_per_block = (block.x + 31) / 32;
        shared_mem_size = 2 * warps_per_block * sizeof(float); // max+sum
    }

    if (shared_mem_size > 0)
    {
        kernel<<<grid, block, shared_mem_size>>>(input, output, size);
    }
    else
    {
        kernel<<<grid, block>>>(input, output, size);
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Kernel execution failed");
}

bool compare_array(float *a, float *b, size_t size, float epsilon = 1e-5f)
{
    float max_diff = 0.0f;
    int mismatch_count = 0;
    const int max_mismatches_to_show = 5;

    for (size_t i = 0; i < size; ++i)
    {
        float diff = fabsf(a[i] - b[i]);
        max_diff = fmaxf(max_diff, diff);

        if (diff > epsilon)
        {
            if (mismatch_count < max_mismatches_to_show)
            {
                std::cerr << "Mismatch at position " << i << ": " << a[i]
                          << " vs " << b[i] << " (diff: " << diff << ")" << std::endl;
            }
            mismatch_count++;
        }
    }

    if (mismatch_count > 0)
    {
        std::cerr << "Total mismatches: " << mismatch_count << "/" << size
                  << ", Max difference: " << max_diff << std::endl;
        return false;
    }

    std::cout << "Arrays match! Max difference: " << max_diff << std::endl;
    return true;
}

void print_arry(float *arr, size_t size)
{
    std::cout << "[ ";
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << arr[i];
        if (i < size - 1)
            std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}

void init_array(float *arr, size_t size)
{
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < size; ++i)
    {
        arr[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 200)) - 100.0f;
    }
}

void soft_max_impl(const char *kernel_name, void (*kernel)(const float *, float *, size_t),
                   const float *h_input, float *h_output, const size_t &size)
{
    std::cout << "\n=== Testing kernel: " << kernel_name << " (size: " << size << ") ===" << std::endl;

    float *d_input = create_device_buffer<float>(size);
    float *d_output = create_device_buffer<float>(size);
    host2device<float>(d_input, h_input, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    soft_max(kernel_name, kernel, d_input, d_output, size);

    cudaEventRecord(start);
    for (int it = 0; it < ITERATIONS; ++it)
    {
        cudaMemset(d_output, 0, size * sizeof(float));
        soft_max(kernel_name, kernel, d_input, d_output, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    device2host<float>(h_output, d_output, size);

    std::vector<float> cpu_output(size);
    soft_max_host(h_input, cpu_output.data(), size);

    auto res = compare_array(h_output, cpu_output.data(), size);
    if (res)
    {
        std::cout << kernel_name << " PASSED! ";
        std::cout << "Average time: " << milliseconds / ITERATIONS << " ms" << std::endl;
    }
    else
    {
        std::cout << kernel_name << " FAILED!" << std::endl;
        if (size <= 20)
        {
            std::cout << "CPU result: ";
            print_arry(cpu_output.data(), size);
            std::cout << "GPU result: ";
            print_arry(h_output, size);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_device_buffer<float>(d_input);
    free_device_buffer<float>(d_output);
}

void test_soft_max()
{
    try
    {
        std::vector<size_t> test_sizes = {32, 100, 256, 1024, 4096};

        for (size_t data_size : test_sizes)
        {
            std::cout << "\n"
                      << std::string(60, '=') << std::endl;
            std::cout << "Testing with data size: " << data_size << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            float *h_input = create_host_buffer<float>(data_size);
            float *h_output = create_host_buffer<float>(data_size);
            init_array(h_input, data_size);

            if (data_size <= 32)
            {
                std::cout << "Input data: ";
                print_arry(h_input, data_size);
            }

            soft_max_impl("soft_max_naive_kernel", soft_max_naive_kernel, h_input, h_output, data_size);
            soft_max_impl("soft_max_reduce_kernel", soft_max_reduce_kernel, h_input, h_output, data_size);
            soft_max_impl("soft_max_vectorized_kernel", soft_max_vectorized_kernel, h_input, h_output, data_size);

            free_host_buffer<float>(h_input);
            free_host_buffer<float>(h_output);
        }

        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "All tests completed!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

int main()
{
    test_soft_max();
}