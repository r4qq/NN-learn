#pragma once

#include "config.hpp"

#include <fstream>
#include <type_traits>


template<typename T>
class Layer
{
    static_assert(std::is_floating_point<T>::value);

    public:
        virtual ~Layer() = default;
        virtual Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) = 0;
        virtual Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) = 0;
        virtual void save(std::ofstream& outFile) = 0;
        virtual void load(std::ifstream& infile) = 0;
};