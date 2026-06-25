#pragma once

#define TYPE_REGULAR 0
#define TYPE_SIMD 1
#define TYPE_SIMD_BLOCK 2
#define TYPE_SIMD_BLOCK_MULTI 3

#define TENSOR_TYPE 3

#if TENSOR_TYPE == 3
#   include "tensor-hpp/Tensor-simd-block-multi.hpp" // IWYU pragma: export
#elif TENSOR_TYPE == 2
#   include "tensor-hpp/Tensor-simd-block.hpp" // IWYU pragma: export
#elif TENSOR_TYPE == 1
#   include "tensor-hpp/Tensor-simd.hpp" // IWYU pragma: export
#elif TENSOR_TYPE == 0
#   include "tensor-hpp/Tensor.hpp" // IWYU pragma: export
#else
#   error invalid tensor type
#endif