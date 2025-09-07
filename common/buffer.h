#ifndef __BUFFER_H__
#define __BUFFER_H__
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include "cuda_runtime.h"

#define CHECK_CUDA_ERROR(err, msg)                                        \
    if (err != cudaSuccess)                                               \
    {                                                                     \
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error(msg);                                    \
    }

template <typename T>
T *create_host_buffer(size_t size)
{
    void *ptr = std::malloc(size * sizeof(T));
    if (!ptr)
        throw std::bad_alloc();
    return static_cast<T *>(ptr);
}

template <typename T>
T *create_device_buffer(size_t size)
{
    T *ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, size * sizeof(T)), "Failed to allocate device memery.");
    return ptr;
}

template <typename T>
void free_host_buffer(T *ptr)
{
    std::free(ptr);
}

template <typename T>
void free_device_buffer(T *ptr)
{
    CHECK_CUDA_ERROR(cudaFree(ptr), "Failed to free device memery.");
}

template <typename T>
void host2device(T *d_mem, const T *h_mem, size_t size)
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_mem, h_mem, size * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy to device");
}

template <typename T>
void device2host(T *h_mem, const T *d_mem, size_t size)
{
    CHECK_CUDA_ERROR(cudaMemcpy(h_mem, d_mem, size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy to host");
}
#endif