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
        Tensor::Tensor<T> _cacheInput;
        Tensor::Tensor<T> _weightGradients;
    
    public:
        DenseLayer(uint64_t inputSize, uint64_t outputSize)
        :   _weights({inputSize, outputSize}),
            _weightGradients({inputSize, outputSize}),
            _biases({outputSize}),
            _cacheInput({1})
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
            _cacheInput = input;
            uint64_t batchSize = input.shape()[0];
            uint64_t outputSize = _weights.shape()[1];

            //input * weights
            Tensor::Tensor<T> outputTensor({batchSize, outputSize});
            Tensor::matmul(input, _weights, outputTensor);
            
            //add biases
            for (uint64_t i = 0; i < batchSize; ++i) 
            {
                for (uint64_t j = 0; j < outputSize; ++j) 
                {
                    outputTensor.unchecked(i, j) += _biases.unchecked(j); 
                }
            }
            return outputTensor;
        }

        Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override
        {
            Tensor::Tensor<T> transposedInput({_cacheInput.shape()[1], _cacheInput.shape()[0]});
            //Tensor::Tensor<T> transposedWeights({})

            //weightsGradient = inputT * outputGradient
            Tensor::transpose(_cacheInput, transposedInput);
            Tensor::matmul(transposedInput, outputGradient, _weightGradients);
            
            _weights -= (_weightGradients * learningRate);
        }
};