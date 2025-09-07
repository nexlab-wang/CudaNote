# CudaNote

CUDA算子优化的学习项目，通过实现和优化常见的GPU计算算子来深入理解CUDA编程和性能优化技术。

## 项目结构

```
CudaNote/
├── common/          # 通用工具模块
│   └── buffer.h     # CUDA内存管理工具
├── Reduce/          # 归约算子实现
├── GEMM/            # 矩阵乘法算子实现
└── Transpose/       # 矩阵转置算子实现
```

## 算子介绍

- **Reduce**: 实现高效的归约操作，包括求和、最大值等
- **GEMM**: 通用矩阵乘法优化实现
- **Transpose**: 矩阵转置算子，重点优化内存访问模式

## 构建说明

```bash
cmake -B build
cmake --build build
```

## 环境要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- 支持CUDA的GPU设备
