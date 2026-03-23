#pragma once

#include "Layer.hpp"
#include <memory>
#include <utility>
#include <vector>
#include <ranges>

template<typename T>
class NeuralNetwork
{
    private:
        std::vector<std::unique_ptr<Layer<T>>> _layers;

    public:
        void addLayer(std::unique_ptr<Layer<T>> layer)
        {
            _layers.push_back(std::move(layer));
        }

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& initInput)
        {
            auto tempTensor = initInput;
            for (const auto& layer : _layers) 
            {
                tempTensor = layer->forward(tempTensor); 
            }

            return tempTensor;
        }

        void backward(const Tensor::Tensor<T>& outputGradient, T learningRate)
        {
            auto tempGrad = outputGradient;
            for (const auto& layer : _layers | std::views::reverse) //some cpp20 shenaningans
            {
                tempGrad = layer->backward(tempGrad, learningRate);
            }
        }
};