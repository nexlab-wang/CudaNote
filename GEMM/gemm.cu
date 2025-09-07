#include "cuda_runtime.h"
#include "common/buffer.h"
#include <vector>
#include <iomanip>

#define ITERATIONS 10


__global__ void gpu_naive_matmul_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    // 每个线程负责计算 C 中的一个元素 (m, n)
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N)
    {
        float val = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            val += A[m * K + k] * B[k * N + n]; // A 是行优先，B 是行优先
        }
        C[m * N + n] = alpha * val + beta * C[m * N + n];
    }
}

__global__ void gpu_shared_memory_matmul_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    __shared__ float A_row[1024]; // 假设 K <= 1024
    __shared__ float B_col[1024]; // 假设 K <= 1024

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float val = 0.0f;

        // 当矩阵大于线程块尺寸，共享内存会冲突，出现错误
        for (int k = 0; k < K; ++k)
        {
            if (threadIdx.x == 0)
            {
                A_row[k] = A[row * K + k];
            }
            if (threadIdx.y == 0)
            {
                B_col[k] = B[k * N + col];
            }
            __syncthreads();

            val += A_row[k] * B_col[k];
            __syncthreads();
        }

        C[row * N + col] = alpha * val + beta * C[row * N + col];
    }
}

template <int BM = 16, int BN = 16, int BK = 16>
__global__ void gpu_tiled_matmul_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    __shared__ float mat_as[BM][BK];
    __shared__ float mat_bs[BK][BN];

    int row = threadIdx.y;
    int col = threadIdx.x;

    int global_row = blockIdx.y * BM + row;
    int global_col = blockIdx.x * BN + col;

    float val = 0.0f;

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t)
    {
        int tiled_k = t * BK;

        if (global_row < M && (tiled_k + col) < K)
        {
            mat_as[row][col] = A[global_row * K + tiled_k + col];
        }
        else
        {
            mat_as[row][col] = 0.0f;
        }

        if ((tiled_k + row) < K && global_col < N)
        {
            mat_bs[row][col] = B[(tiled_k + row) * N + global_col];
        }
        else
        {
            mat_bs[row][col] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k)
        {
            val += mat_as[row][k] * mat_bs[k][col];
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N)
    {
        C[global_row * N + global_col] = alpha * val + beta * C[global_row * N + global_col];
    }
}

template <int BM = 32, int BN = 32, int BK = 32, int TM = 2, int TN = 2>
__global__ void gpu_thread_tiled_matmul_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    __shared__ float mat_as[BM][BK];
    __shared__ float mat_bs[BK][BN];

    float reg_a[TM];
    float reg_b[TN];
    float acc[TM][TN] = {0.0f};

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int global_row = blockIdx.y * BM + thread_row * TM;
    int global_col = blockIdx.x * BN + thread_col * TN;

    // 每个线程要现有负责的数据量
    int sub_row = BK / blockDim.y; // 确定b矩阵y方向一个矩阵tile与线程block的对应关系
    int sub_col = BK / blockDim.x; // 确定a矩阵x方向一个矩阵tile与线程block的对应关系

    int num_tiles = (K + BK - 1) / BK;

#pragma unroll
    for (int t = 0; t < num_tiles; ++t)
    {

        int tiled_k = t * BK; // 控制矩阵tile在BK维度上的迭代

#pragma unroll
        for (int i = 0; i < TM; ++i)
        {

#pragma unroll
            for (int c = 0; c < sub_col; ++c)
            {
                int a_row = thread_row * TM + i;
                int a_col = thread_col * sub_col + c;
                int load_row = global_row + i;
                int load_col = tiled_k + thread_col * sub_col + c;

                mat_as[a_row][a_col] = (load_row < M && load_col < K) ? A[load_row * K + load_col] : 0.0f;
            }
        }

#pragma unroll
        for (int j = 0; j < TN; ++j)
        {

#pragma unroll
            for (int r = 0; r < sub_row; ++r)
            {
                int b_row = thread_row * sub_row + r;
                int b_col = thread_col * TN + j;
                int load_row = tiled_k + thread_row * sub_row + r;
                int load_col = global_col + j;

                mat_bs[b_row][b_col] = (load_row < K && load_col < N) ? B[load_row * N + load_col] : 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k)
        {

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                reg_a[i] = mat_as[thread_row * TM + i][k];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    reg_b[j] = mat_bs[k][thread_col * TN + j];
                }
            }

#pragma unroll
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += reg_a[i] * reg_b[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {

#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            if ((global_row + i) < M && (global_col + j) < N)
            {
                C[(global_row + i) * N + (global_col + j)] = alpha * acc[i][j] + beta * C[(global_row + i) * N + (global_col + j)];
            }
        }
    }
}

template <int BM = 64, int BN = 64, int BK = 32, int TM = 4, int TN = 4, int VEC_SIZE = 4>
__global__ void gpu_matmul_float4_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;

    const int block_col = blockIdx.x * BN;
    const int block_row = blockIdx.y * BM;

    __shared__ float mat_as[BM][BK];
    __shared__ float mat_bs[BK][BN];

    float acc[TM][TN] = {0.0f};

    constexpr int WPTN = TN / VEC_SIZE; // 列方向被float4加载全局内存后，每个线程负责数据块的个数

    const int num_tiles = (K + BK - 1) / BK;

#pragma unroll
    for (int t = 0; t < num_tiles; ++t)
    {
        int tiled_k = t * BK;

        // load shared A
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            int a_row = block_row + thread_row * TM + i;
            int shared_a_row = thread_row * TM + i;
            // 行优先，使用float4访问4个连续元素
#pragma unroll
            for (int k = 0; k < BK; k += VEC_SIZE)
            {
                int a_col = tiled_k + k;
                float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                if (a_row < M && a_col + 3 < K)
                {
                    vec = *(const float4 *)(&A[a_row * K + a_col]);
                }

                mat_as[shared_a_row][k + 0] = vec.x;
                mat_as[shared_a_row][k + 1] = vec.y;
                mat_as[shared_a_row][k + 2] = vec.z;
                mat_as[shared_a_row][k + 3] = vec.w;
            }
        }

        // load shared B
#pragma unroll
        for (int j = 0; j < WPTN; ++j)
        {
            int b_col = block_col + thread_col * TN + j * VEC_SIZE;
            // 行优先存储，列优先访问，固定间距的访问同一列四个数
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                int b_row = tiled_k + k;
                float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                if (b_row < K && b_col + 3 < N)
                {
                    vec = *(const float4 *)(&B[b_row * N + b_col]);
                }

                mat_bs[k][thread_col * TN + j * VEC_SIZE + 0] = vec.x;
                mat_bs[k][thread_col * TN + j * VEC_SIZE + 1] = vec.y;
                mat_bs[k][thread_col * TN + j * VEC_SIZE + 2] = vec.z;
                mat_bs[k][thread_col * TN + j * VEC_SIZE + 3] = vec.w;
            }
        }
        __syncthreads();

        // compute
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            float a_reg[TM];
            float b_reg[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                a_reg[i] = mat_as[thread_row * TM + i][k];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                b_reg[j] = mat_bs[k][thread_col * TN + j];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {

#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // write c
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int c_row = block_row + thread_row * TM + i;
        if (c_row >= M)
            continue;

#pragma unroll
        for (int j = 0; j < WPTN; ++j)
        {
            int c_col = block_col + thread_col * TN + j * VEC_SIZE;
            if (c_col + 3 > N)
                continue;

            float4 c_old = *(const float4 *)(&C[c_row * N + c_col]);
            float4 c_new;
            c_new.x = alpha * acc[i][j * VEC_SIZE + 0] + beta * c_old.x;
            c_new.y = alpha * acc[i][j * VEC_SIZE + 1] + beta * c_old.y;
            c_new.z = alpha * acc[i][j * VEC_SIZE + 2] + beta * c_old.z;
            c_new.w = alpha * acc[i][j * VEC_SIZE + 3] + beta * c_old.w;

            *(float4 *)(&C[c_row * N + c_col]) = c_new;
        }
    }
}

template <int BM = 64, int BN = 64, int BK = 32, int TM = 4, int TN = 4, int VEC_SIZE = 4>
__global__ void gpu_matmul_float4_optimized_kernel(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;

    const int block_col = blockIdx.x * BN;
    const int block_row = blockIdx.y * BM;

    // B矩阵转置存储 [BN][BK] -> 列优先
    __shared__ float mat_as[BM][BK];
    __shared__ float mat_bs[BN][BK]; // 转置后的布局

    float acc[TM][TN] = {0.0f};
    constexpr int WPTN = TN / VEC_SIZE;
    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t)
    {
        int tiled_k = t * BK;

// A矩阵加载保持不变（行优先）
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            int a_row = block_row + thread_row * TM + i;
            int shared_a_row = thread_row * TM + i;
#pragma unroll
            for (int k = 0; k < BK; k += VEC_SIZE)
            {
                int a_col = tiled_k + k;
                float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (a_row < M && a_col + 3 < K)
                {
                    vec = *(const float4 *)(&A[a_row * K + a_col]);
                }
                mat_as[shared_a_row][k + 0] = vec.x;
                mat_as[shared_a_row][k + 1] = vec.y;
                mat_as[shared_a_row][k + 2] = vec.z;
                mat_as[shared_a_row][k + 3] = vec.w;
            }
        }

// B矩阵加载改为列优先存储
#pragma unroll
        for (int j = 0; j < WPTN; ++j)
        {
            int b_col = block_col + thread_col * TN + j * VEC_SIZE;
#pragma unroll
            for (int k = 0; k < BK; ++k)
            {
                int b_row = tiled_k + k;
                float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (b_row < K && b_col + 3 < N)
                {
                    vec = *(const float4 *)(&B[b_row * N + b_col]);
                }
                // 转置存储：行列交换
                mat_bs[thread_col * TN + j * VEC_SIZE + 0][k] = vec.x;
                mat_bs[thread_col * TN + j * VEC_SIZE + 1][k] = vec.y;
                mat_bs[thread_col * TN + j * VEC_SIZE + 2][k] = vec.z;
                mat_bs[thread_col * TN + j * VEC_SIZE + 3][k] = vec.w;
            }
        }
        __syncthreads();

// 计算时访问转置后的B矩阵
#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            float a_reg[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                a_reg[i] = mat_as[thread_row * TM + i][k];
            }

            float b_reg[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                // 行优先访问转置后的B矩阵
                b_reg[j] = mat_bs[thread_col * TN + j][k];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

// 结果写入保持不变
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int c_row = block_row + thread_row * TM + i;
        if (c_row >= M)
            continue;
#pragma unroll
        for (int j = 0; j < WPTN; ++j)
        {
            int c_col = block_col + thread_col * TN + j * VEC_SIZE;
            if (c_col + 3 > N)
                continue;
            float4 c_old = *(const float4 *)(&C[c_row * N + c_col]);
            float4 c_new;
            c_new.x = alpha * acc[i][j * VEC_SIZE + 0] + beta * c_old.x;
            c_new.y = alpha * acc[i][j * VEC_SIZE + 1] + beta * c_old.y;
            c_new.z = alpha * acc[i][j * VEC_SIZE + 2] + beta * c_old.z;
            c_new.w = alpha * acc[i][j * VEC_SIZE + 3] + beta * c_old.w;
            *(float4 *)(&C[c_row * N + c_col]) = c_new;
        }
    }
}

void cpu_naive_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float val = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                val += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * val + beta * C[m * N + n];
        }
    }
}

void gemm(const std::string &kernel_name, void (*kernel)(int, int, int, float, const float *, const float *, float, float *),
                     const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{

    dim3 block;
    dim3 grid;

    if (kernel_name == "gpu_naive_matmul_kernel")
    {
        block = dim3(16, 16);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "gpu_tiled_matmul_kernel")
    {
        block = dim3(16, 16);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "gpu_thread_tiled_matmul_kernel")
    {
        block = dim3(16, 16);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "gpu_matmul_float4_kernel")
    {
        block = dim3(16, 16);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }
    if (kernel_name == "gpu_matmul_float4_optimized_kernel")
    {
        block = dim3(16, 16);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    }

    kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Kernel execution failed");
}

void init_matrix(float *matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    }
}

void init_matrix(float *matrix, int rows, int cols, bool is_A)
{
    if (is_A)
    {
        // Initialize A as 1,2,3,...m; m+1,m+2,...2m; .
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i * cols + j] = i * cols + j + 1;
            }
        }
    }
    else
    {
        // Initialize B as 1,n+1,2n+1,...; 2,n+2,2n+2,...; .
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i * cols + j] = j * rows + i + 1; 
            }
        }
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

void print_matrix(const float *matrix,
                  int rows,
                  int cols,
                  int precision = 4,
                  int width = 10,
                  int max_rows = -1,
                  int max_cols = -1)
{
    if (max_rows <= 0 || max_rows > rows)
        max_rows = rows;
    if (max_cols <= 0 || max_cols > cols)
        max_cols = cols;

    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);

    std::cout << std::fixed << std::setprecision(precision);

    // 打印矩阵
    for (int i = 0; i < max_rows; ++i)
    {
        std::cout << (i == 0 ? "[" : " ");

        for (int j = 0; j < max_cols; ++j)
        {
            std::cout << std::setw(width) << matrix[i * cols + j];

            if (j == max_cols - 1 && max_cols < cols)
            {
                std::cout << std::setw(width) << "...";
            }
        }

        if (i == max_rows - 1 && max_rows < rows)
        {
            std::cout << "\n ...";
        }

        std::cout << (i == max_rows - 1 ? "]" : "\n");
    }

    // 恢复原始输出格式
    std::cout.copyfmt(old_state);
    std::cout << "\n";
}

void gemm_impl(const char *kernel_name, void (*kernel)(int, int, int, float, const float *, const float *, float, float *),
               const float *mat_a, const float *mat_b, float *mat_c, int M, int N, int K, float alpha, float beta)
{
    std::cout << "\n=== Testing kernel: " << kernel_name << " ===" << std::endl;

    size_t mat_a_size = M * K;
    size_t mat_b_size = K * N;
    size_t mat_c_size = M * N;
    float *d_mat_a = create_device_buffer<float>(mat_a_size);
    float *d_mat_b = create_device_buffer<float>(mat_b_size);
    float *d_mat_c = create_device_buffer<float>(mat_c_size);

    host2device<float>(d_mat_a, mat_a, mat_a_size);
    host2device<float>(d_mat_b, mat_b, mat_b_size);

    // 设置循环迭代

    for (int it = 0; it < ITERATIONS; ++it)
    {
        memset(mat_c, 0, mat_c_size * sizeof(float));
        cudaMemset(d_mat_c, 0, mat_c_size * sizeof(float));

        gemm(kernel_name, kernel, d_mat_a, d_mat_b, d_mat_c, M, N, K, alpha, beta);
    }

    device2host<float>(mat_c, d_mat_c, mat_c_size);

    std::vector<float> cpu_mat_c(mat_c_size);
    cpu_naive_matmul(M, N, K, alpha, mat_a, mat_b, beta, cpu_mat_c.data());

    compare_matrixes(mat_c, cpu_mat_c.data(), mat_c_size);
    // print_matrix(mat_c,M,N);
    // print_matrix(cpu_mat_c.data(),M,N);

    free_device_buffer<float>(d_mat_a);
    free_device_buffer<float>(d_mat_b);
    free_device_buffer<float>(d_mat_c);
}

void test_gemm()
{
    try
    {
        int M = 128;
        int N = 128;
        int K = 128;
        float alpha = 1.0f;
        float beta = 0.5f;

        size_t mat_a_size = M * K;
        size_t mat_b_size = K * N;
        size_t mat_c_size = M * N;

        float *h_mat_a = create_host_buffer<float>(mat_a_size);
        float *h_mat_b = create_host_buffer<float>(mat_b_size);
        float *h_mat_c = create_host_buffer<float>(mat_c_size);

        init_matrix(h_mat_a, mat_a_size);
        init_matrix(h_mat_b, mat_b_size);
        // init_matrix(h_mat_a, M, K, true);  // Initialize A
        // init_matrix(h_mat_b, K, N, false); // Initialize B

        gemm_impl("gpu_naive_matmul_kernel", gpu_naive_matmul_kernel,
                  h_mat_a, h_mat_b, h_mat_c, M, N, K, alpha, beta);

        // // gemm_impl("gpu_shared_memory_matmul_kernel", gpu_shared_memory_matmul_kernel,
        // //           h_mat_a, h_mat_b, h_mat_c, M, N, K, alpha, beta);

        gemm_impl("gpu_tiled_matmul_kernel", gpu_tiled_matmul_kernel,
                  h_mat_a, h_mat_b, h_mat_c, M, N, K, alpha, beta);

        gemm_impl("gpu_thread_tiled_matmul_kernel", gpu_thread_tiled_matmul_kernel,
                  h_mat_a, h_mat_b, h_mat_c, M, N, K, alpha, beta);

        gemm_impl("gpu_matmul_float4_kernel", gpu_matmul_float4_kernel,
                  h_mat_a, h_mat_b, h_mat_c, M, N, K, alpha, beta);

        gemm_impl("gpu_matmul_float4_optimized_kernel", gpu_matmul_float4_optimized_kernel,
                  h_mat_a, h_mat_b, h_mat_c, M, N, K, alpha, beta);

        free_host_buffer<float>(h_mat_a);
        free_host_buffer<float>(h_mat_b);
        free_host_buffer<float>(h_mat_c);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

int main()
{
    test_gemm();
    return 0;
}
