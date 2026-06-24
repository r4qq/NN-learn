#pragma once

#include <cstdint>

enum class LayerType : uint32_t
{
    Dense = 1,
    ReLu = 2,
    Sigmoid = 3,
    Softmax = 4,
    Tanh = 5
};