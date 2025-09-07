#include <cuda_runtime.h>
#include "common/buffer.h"
#include <iostream>
#include <vector>

#define ITERATIONS 10

void trans_cpu(const float *A, float *B, int M, int N)
{
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            B[n * M + m] = A[m * N + n];
        }
    }
}

__global__ void naive_trans_kernel(const float *A, float *B, int M, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x ∈ [0, N)，列方向
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y ∈ [0, M)，行方向

    if (x < N && y < M)
    {
        B[x * M + y] = A[y * N + x];
    }
}

template <int BM = 32, int BN = 8>
__global__ void tiled_trans_kernel(const float *A, float *B, int M, int N)
{
    __shared__ float tile[BN][BM]; // 转置写入

    int col = blockIdx.x * BN + threadIdx.x;
    int row = blockIdx.y * BM + threadIdx.y;

    if (col < N && row < M)
    {
        tile[threadIdx.x][threadIdx.y] = A[row * N + col];
    }
    __syncthreads();

    int trans_col = blockIdx.y * BM + threadIdx.y;
    int trans_row = blockIdx.x * BN + threadIdx.x;

    if (trans_col < M && trans_row < N)
    {
        B[trans_row * M + trans_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int BM = 64, int BN = 32, int TM = 2, int TN = 4>
__global__ void thread_tiled_trans_kernel(const float *A, float *B, int M, int N)
{
    __shared__ float tile[BN][BM]; // 转置写入

    int bolck_col = blockIdx.x * (BN);
    int block_row = blockIdx.y * (BM);

#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
#pragma unroll
        for (int n = 0; n < TN; ++n)
        {
            int col = bolck_col + threadIdx.x * TN + n;
            int row = block_row + threadIdx.y * TM + m;
            if (col < N && row < M)
            {
                tile[threadIdx.x * TN + n][threadIdx.y * TM + m] = A[row * N + col];
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
#pragma unroll
        for (int n = 0; n < TN; ++n)
        {
            int trans_col = block_row + threadIdx.y * TM + m;
            int trans_row = bolck_col + threadIdx.x * TN + n;
            if (trans_col < M && trans_row < N)
            {
                B[trans_row * M + trans_col] = tile[threadIdx.x * TN + n][threadIdx.y * TM + m];
            }
        }
    }
}

template <int BM = 64, int BN = 32, int TM = 2, int TN = 4>
__global__ void swizzling_trans_kernel(const float *A, float *B, int M, int N)
{
    __shared__ float tile[BN][BM]; // 转置写入

    int bolck_col = blockIdx.x * (BN);
    int block_row = blockIdx.y * (BM);

#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
#pragma unroll
        for (int n = 0; n < TN; ++n)
        {
            int col = bolck_col + threadIdx.x * TN + n;
            int row = block_row + threadIdx.y * TM + m;
            if (col < N && row < M)
            {
                //XOR
                int swizzled_col = (threadIdx.x * TN + n) ^ (threadIdx.y * TM + m);
                int swizzled_row = threadIdx.y*TM+m;

                swizzled_col %= BN;
                swizzled_row %= BM;
                tile[swizzled_col][swizzled_row] = A[row * N + col];
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
#pragma unroll
        for (int n = 0; n < TN; ++n)
        {
            int trans_col = block_row + threadIdx.y * TM + m;
            int trans_row = bolck_col + threadIdx.x * TN + n;
            if (trans_col < M && trans_row < N)
            {
                int original_col = (threadIdx.x * TN + n) ^ (threadIdx.y * TM + m);
                int original_row = threadIdx.y * TM + m;

                original_col %= BN;
                original_row %= BM;
                B[trans_row * M + trans_col] = tile[original_col][original_row];
            }
        }
    }
}

void transpose(const std::string &kernel_name, void (*kernel)(const float *, float *, int, int),
               const float *A, float *B, int M, int N)
{
    dim3 block;
    dim3 grid;
    if (kernel_name == "naive_trans_kernel")
    {
        block = dim3(4, 64);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "tiled_trans_kernel")
    {
        block = dim3(8, 32);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "thread_tiled_trans_kernel")
    {
        block = dim3(8, 32);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "swizzling_trans_kernel")
    {
        block = dim3(8, 32);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    kernel<<<grid, block>>>(A, B, M, N);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Kernel execution failed");
}

void init_matrix(float *matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    }
}

bool compare_matrixes(const float *mat_a, const float *mat_b, int size, float epsilon = 1e-4f)
{
    for (int i = 0; i < size; ++i)
    {
        if (abs(mat_a[i] - mat_b[i]) > epsilon)
        {
            std::cerr << "Mismatch at position " << i << ": " << mat_a[i] << " vs " << mat_b[i] << std::endl;
            return false;
        }
    }

    return true;
}

void transpose_impl(const char *kernel_name, void (*kernel)(const float *, float *, int, int),
                    const float *mat_a, float *mat_b, int M, int N)
{
    std::cout << "\n=== Testing kernel: " << kernel_name << " ===" << std::endl;

    size_t mat_a_size = M * N;
    size_t mat_b_size = N * M;
    float *d_mat_a = create_device_buffer<float>(mat_a_size);
    float *d_mat_b = create_device_buffer<float>(mat_b_size);

    host2device<float>(d_mat_a, mat_a, mat_a_size);
    host2device<float>(d_mat_b, mat_b, mat_b_size);

    for (int it = 0; it < ITERATIONS; ++it)
    {
        cudaMemset(d_mat_b, 0, mat_b_size * sizeof(float));

        transpose(kernel_name, kernel, d_mat_a, d_mat_b, M, N);
    }
    device2host<float>(mat_b, d_mat_b, mat_b_size);

    std::vector<float> cpu_mat_b(mat_b_size);
    trans_cpu(mat_a, cpu_mat_b.data(), M, N);

    compare_matrixes(mat_b, cpu_mat_b.data(), mat_b_size);

    free_device_buffer<float>(d_mat_a);
    free_device_buffer<float>(d_mat_b);
}

void test_transpose()
{
    try
    {
        int M = 1024;
        int N = 1024;

        size_t mat_a_size = M * N;
        size_t mat_b_size = N * M;

        float *h_mat_a = create_host_buffer<float>(mat_a_size);
        float *h_mat_b = create_host_buffer<float>(mat_b_size);

        init_matrix(h_mat_a, mat_a_size);
        init_matrix(h_mat_b, mat_b_size);

        transpose_impl("naive_trans_kernel", naive_trans_kernel, h_mat_a, h_mat_b, M, N);

        transpose_impl("tiled_trans_kernel", tiled_trans_kernel, h_mat_a, h_mat_b, M, N);

        transpose_impl("thread_tiled_trans_kernel", thread_tiled_trans_kernel, h_mat_a, h_mat_b, M, N);

        transpose_impl("swizzling_trans_kernel", swizzling_trans_kernel, h_mat_a, h_mat_b, M, N);

        free_host_buffer<float>(h_mat_a);
        free_host_buffer<float>(h_mat_b);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

int main()
{
    test_transpose();
    return 0;
}