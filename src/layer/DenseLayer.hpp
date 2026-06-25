#pragma once

#include "core/Layer.hpp"
#include "LayerType.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <fstream>
#include <sys/types.h>

namespace NN::Layers 
{
    template<typename T>
    class DenseLayer : public NN::Core::Layer<T>
    {
        private:
            Tensor::Tensor<T> _weights;
            Tensor::Tensor<T> _biases;
            Tensor::Tensor<T> _cacheInput;
            Tensor::Tensor<T> _weightGradients;
            Tensor::Tensor<T> _cachedOutput;
            Tensor::Tensor<T> _transposedInput;
            Tensor::Tensor<T> _transposedWeights;
            Tensor::Tensor<T> _cachedInputGrad;
            Tensor::Tensor<T> _biasGradients;    

        public:
            DenseLayer(uint64_t inputSize, uint64_t outputSize)
            :   _weights({inputSize, outputSize}),
                _biases({outputSize}),
                _cacheInput({1}),
                _weightGradients({inputSize, outputSize}),
                _cachedOutput({1}),
                _transposedInput({1}),
                _transposedWeights({1}),
                _cachedInputGrad({1}),
                _biasGradients({outputSize})
                
            {
                _biases.fill(T{0});
                _biasGradients(T{0});

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

                
                std::vector<uint64_t> expectedShape = {batchSize, outputSize};
                if (_cachedOutput.shape() != expectedShape) [[unlikely]]
                {
                    _cachedOutput = Tensor::Tensor<T>(expectedShape);
                }            
                
                //input * weights
                Tensor::matmul(input, _weights, _cachedOutput);
                
                //add biases
                for (uint64_t i = 0; i < batchSize; ++i) 
                {
                    for (uint64_t j = 0; j < outputSize; ++j) 
                    {
                        _cachedOutput.unchecked(i, j) += _biases.unchecked(j); 
                    }
                }
                return _cachedOutput;
            }

            Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override
            {
                uint64_t batchSize = outputGradient.shape()[0];
                uint64_t inputSize = _weights.shape()[0];
                uint64_t outputSize = _weights.shape()[1];

                std::vector<uint64_t> expectedTransInputShape = {_cacheInput.shape()[1], _cacheInput.shape()[0]};
                if (_transposedInput.shape() != expectedTransInputShape) [[unlikely]]
                {
                    _transposedInput = Tensor::Tensor<T>(expectedTransInputShape);
                }

                std::vector<uint64_t> expectedTransWeightShape = {_weights.shape()[1], _weights.shape()[0]};
                if (_transposedWeights.shape() != expectedTransWeightShape) [[unlikely]]
                {
                    _transposedWeights = Tensor::Tensor<T>(expectedTransWeightShape);
                }

                std::vector<uint64_t> expectedInputGradShape = {batchSize, inputSize};
                if (_cachedInputGrad.shape() != expectedInputGradShape) [[unlikely]]
                {
                    _cachedInputGrad = Tensor::Tensor<T>(expectedInputGradShape);
                }

                //weightsGradient = inputT * outputGradient
                Tensor::transpose(_cacheInput, _transposedInput);
                Tensor::matmul(_transposedInput, outputGradient, _weightGradients);
                
                //inputGradient = weightsT * outputGradient
                Tensor::transpose(_weights, _transposedWeights);
                Tensor::matmul(outputGradient, _transposedWeights, _cachedInputGrad);

                //biasGradient
                _biasGradients.fill(T{0});

                for (uint64_t i = 0; i < batchSize; ++i) 
                {
                    for (uint64_t j = 0; j < outputSize; ++j) 
                    {
                        _biasGradients.unchecked(j) += outputGradient.unchecked(i, j); 
                    }
                }

                //update weights adn biases
                _weights -= (_weightGradients * learningRate);
                _biases -= (_biasGradients * learningRate);

                return _cachedInputGrad;
            }

            void save(std::ofstream& outFile) override 
            {
                LayerType type = LayerType::Dense;
                outFile.write(reinterpret_cast<const char*>(&type), sizeof(uint32_t));

                uint64_t inSize = _weights.shape()[0];
                uint64_t outSize = _weights.shape()[1];
                outFile.write(reinterpret_cast<const char*>(&inSize), sizeof(uint64_t));
                outFile.write(reinterpret_cast<const char*>(&outSize), sizeof(uint64_t));
                
                outFile.write(reinterpret_cast<const char*>(_weights.data()), _weights.size() * sizeof(T));
                outFile.write(reinterpret_cast<const char*>(_biases.data()), _biases.size() * sizeof(T));
            };

            void load(std::ifstream& infile) override 
            {
                infile.read(reinterpret_cast<char*>(_weights.data()), _weights.size() * sizeof(T));
                infile.read(reinterpret_cast<char*>(_biases.data()), _biases.size() * sizeof(T));
            }
    };
}