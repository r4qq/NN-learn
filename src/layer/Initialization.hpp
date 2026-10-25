#pragma once

#include "core/config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>

namespace NN::Layers::Initialization 
{
    template<typename T>
    using InitFunc = std::function<void(Tensor::Tensor<T>& inputTensor, uint64_t inSize, uint64_t outsize)>;

    template<typename T>
    void heNormal(Tensor::Tensor<T>& inputTensor, uint64_t inSize, [[maybe_unused]] uint64_t outsize)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        T stdDev = std::sqrt(static_cast<T>(2) / static_cast<T>(inSize));

        std::normal_distribution<T> dist(T{0}, stdDev);

        std::generate_n(inputTensor.data(), inputTensor.size(), [&](){return dist(gen);});
    }

    template<typename T>
    void xavierUniform(Tensor::Tensor<T>& tensor, uint64_t insize, uint64_t outSize) 
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        T limit = std::sqrt(static_cast<T>(6.0) / static_cast<T>(insize + outSize));
        
        std::uniform_real_distribution<T> dist(-limit, limit);

        std::generate_n(tensor.data(), tensor.size(), [&]() { return dist(gen); });
    }

    template<typename T>
    void zero(Tensor::Tensor<T>& inputTensor, [[maybe_unused]] uint64_t inSize, [[maybe_unused]] uint64_t outsize)
    {
        inputTensor.fill(T{0});
    }
}