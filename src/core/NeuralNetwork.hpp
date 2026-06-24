#pragma once

#include "config.hpp"
#include <string>

namespace NN::Core 
{
    template<typename T>
    class NeuralNetwork 
    {
        public:
            virtual ~NeuralNetwork() = default;
            virtual Tensor::Tensor<T> forward(const Tensor::Tensor<T>& initInput) = 0;
            virtual void backward(const Tensor::Tensor<T>& outputGradient, T learningRate) = 0;
            virtual void save(std::string fileName) = 0;
            virtual void load(std::string fileName) = 0;
    };
}
