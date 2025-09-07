#include "cuda_runtime.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>

#include "common/buffer.h"

#define BLOCK_SIZE 256
#define ITERATIONS 100
#define ERROR_TOLERANCE 1e-5f


// Kernel declarations
__global__ void reduce_kernel_baseline(const float *data, const size_t n, float *result)
{

    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1.块内归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2.块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}

__global__ void reduce_kernel_warp_divergent(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1.块内归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        // if (tid & (2 * s - 1) == 0)
        // {
        //     sdata[tid] += sdata[tid + s];
        // }
        __syncthreads();
    }

    // 2.块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}

__global__ void reduce_kernel_bank_conflict(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1.块内归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2.块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}

__device__ void warp_reduce(volatile float *sdata, int tid)
{
    if (BLOCK_SIZE >= 64)
        sdata[tid] += sdata[tid + 32];
    if (BLOCK_SIZE >= 32)
        sdata[tid] += sdata[tid + 16];
    if (BLOCK_SIZE >= 16)
        sdata[tid] += sdata[tid + 8];
    if (BLOCK_SIZE >= 8)
        sdata[tid] += sdata[tid + 4];
    if (BLOCK_SIZE >= 4)
        sdata[tid] += sdata[tid + 2];
    if (BLOCK_SIZE >= 2)
        sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_kernel_warp(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1.块内归约(s>32,多warp活跃，需要同步)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 1.1 s<32,1g个warp活跃，无需同步
    if (tid < 32)
        warp_reduce(sdata, tid);

    // 2.块内归约
    if (tid == 0)
        atomicAdd(&result[0], sdata[0]);
}

__global__ void reduce_kernel_loop_unrolling(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    if (BLOCK_SIZE >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
        warp_reduce(sdata, tid);

    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}

__inline__ __device__ void warp_shuffle(float &sum)
{
    unsigned int FULL_MASK = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    // if (BLOCK_SIZE >= 32)
    // {
    //     sum += __shfl_down_sync(FULL_MASK, sum, 16);
    // }
    // if (BLOCK_SIZE >= 16)
    // {
    //     sum += __shfl_down_sync(FULL_MASK, sum, 8);
    // }
    // if (BLOCK_SIZE >= 8)
    // {
    //     sum += __shfl_down_sync(FULL_MASK, sum, 4);
    // }
    // if (BLOCK_SIZE >= 4)
    // {
    //     sum += __shfl_down_sync(FULL_MASK, sum, 2);
    // }
    // if (BLOCK_SIZE >= 2)
    // {
    //     sum += __shfl_down_sync(FULL_MASK, sum, 1);
    // }
}

__global__ void reduce_kernel_warp_shuffle(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    if (BLOCK_SIZE >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 64)
    {
        if (tid < 32)
        {
            sdata[tid] += sdata[tid + 32];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        float sum = sdata[tid];
        warp_shuffle(sum);
        if (tid == 0)
        {
            atomicAdd(&result[0], sum);
        }
    }
}

__global__ void reduce_kernel_thread_coarsening(const float *data, const size_t n, float *result)
{

    __shared__ float sdata[BLOCK_SIZE];

    unsigned int grid_size = BLOCK_SIZE * gridDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = 0;
    float sum = sdata[tid];

    while (global_id < n)
    {
        sum += data[global_id];
        global_id += grid_size;
    }
    sdata[tid] = sum;

    __syncthreads();

    if (BLOCK_SIZE >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 64)
    {
        if (tid < 32)
        {
            sdata[tid] += sdata[tid + 32];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        float sum = sdata[tid];
        warp_shuffle(sum);
        if (tid == 0)
        {
            atomicAdd(&result[0], sum);
        }
    }
}

void initialize_array(float *array, size_t size);

float cpu_reduction(const float *array, size_t size);

bool verify_result(float cpu_result, float gpu_result, float tolerance = ERROR_TOLERANCE);

void reduction(void (*kernel)(const float *, const size_t, float *),
               const float *input, float *output, const size_t size)
{
    size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel<<<num_blocks, BLOCK_SIZE>>>(input, size, output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Kernel execution failed");
}

void reduction_impl(const char *kernel_name, void (*kernel)(const float *, const size_t, float *),
                    const float *h_input, const float *d_input, float *d_output, const size_t size)
{
    std::cout << "\n=== Testing kernel: " << kernel_name << " ===" << std::endl;

    for (int it = 0; it < ITERATIONS; ++it)
    {
        cudaMemset(d_output, 0, sizeof(float));
        reduction(kernel, d_input, d_output, size);
    }

    std::vector<float> h_output(1);
    device2host<float>(h_output.data(), d_output, 1);

    float cpu_result = cpu_reduction(h_input, size);

    // Verify results
    if (!verify_result(cpu_result, h_output.front()))
    {
        std::cerr << "  WARNING: Results do not match within tolerance!\n";
    }
}

void test_reduction()
{
    srand(static_cast<unsigned>(time(nullptr)));
    try
    {
        int array_size = 1000;
        // 1. Prepare host data (keep original copy)
        float *h_input = create_host_buffer<float>(array_size);
        initialize_array(h_input, array_size);

        // 2. Prepare device memory
        float *d_input = create_device_buffer<float>(array_size);
        float *d_output = create_device_buffer<float>(1);
        host2device<float>(d_input, h_input, array_size);

        // 3. launch kernel
        reduction_impl("reduce_kernel_baseline", reduce_kernel_baseline, h_input, d_input, d_output, array_size);
        reduction_impl("reduce_kernel_warp_divergent", reduce_kernel_warp_divergent, h_input, d_input, d_output, array_size);
        reduction_impl("reduce_kernel_bank_conflict", reduce_kernel_bank_conflict, h_input, d_input, d_output, array_size);
        reduction_impl("reduce_kernel_warp", reduce_kernel_warp, h_input, d_input, d_output, array_size);
        reduction_impl("reduce_kernel_loop_unrolling", reduce_kernel_loop_unrolling, h_input, d_input, d_output, array_size);
        reduction_impl("reduce_kernel_warp_shuffle", reduce_kernel_warp_shuffle, h_input, d_input, d_output, array_size);
        reduction_impl("reduce_kernel_thread_coarsening", reduce_kernel_thread_coarsening, h_input, d_input, d_output, array_size);

        // 4. cleanup
        free_host_buffer<float>(h_input);
        free_device_buffer<float>(d_input);
        free_device_buffer<float>(d_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

int main()
{
    test_reduction();
    return 0;
}

void initialize_array(float *array, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = static_cast<float>(rand()) / RAND_MAX; // [0, 1]
    }
}

float cpu_reduction(const float *array, size_t size)
{
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i)
    {
        sum += array[i];
    }
    return static_cast<float>(sum);
}

bool verify_result(float cpu_result, float gpu_result, float tolerance)
{
    float diff = fabs(cpu_result - gpu_result);
    float rel_diff = diff / fmaxf(fabs(cpu_result), 1.0f);

    if (diff > tolerance && rel_diff > tolerance)
    {
        std::cerr << "  Verification failed:\n"
                  << "  CPU result: " << cpu_result << "\n"
                  << "  GPU result: " << gpu_result << "\n"
                  << "  Absolute difference: " << diff << "\n"
                  << "  Relative difference: " << rel_diff * 100.0f << "%\n";
        return false;
    }
    return true;
}