#pragma once

#include "Layer.hpp"
#include <algorithm>
#include <cstdint>
#include <random>

template<typename T>
class DenseLayer : public Layer<T>
{
    private:
        Tensor::Tensor<T> _weights;
        Tensor::Tensor<T> _biases;
    
    public:
        DenseLayer(uint64_t inputSize, uint64_t outputSize)
        :   _weights({inputSize, outputSize}),
            _biases({outputSize})

        {
 
            _biases.fill(T{0});

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(0.0, 0.1);

            std::generate_n(_weights.data(), 
                            _weights.size(), 
                            [&dist, &gen](){
                                return dist(gen);
                            });

        }

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) override
        {
            return input;
        }

        Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override
        {
            return outputGradient * learningRate;
        }
};