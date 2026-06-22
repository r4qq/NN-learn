#pragma once

#include "Layer.hpp"
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ranges>

template<typename T>
class NeuralNetwork {
    public:
        virtual ~NeuralNetwork<T>() = default;
        virtual Tensor::Tensor<T> forward(const Tensor::Tensor<T>& initInput) = 0;
        virtual void backward(const Tensor::Tensor<T>& outputGradient, T learningRate) = 0;
        virtual void save(std::string fileName) = 0;
        virtual void load(std::string fileName) = 0;
};

namespace NN {
    template<typename T>
    class MLP : NeuralNetwork<T>
    {
    private:
        std::vector<std::unique_ptr<Layer<T>>> _layers;

    public:
        void addLayer(std::unique_ptr<Layer<T>> layer)
        {
            _layers.push_back(std::move(layer));
        }

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& initInput) override
        {
            auto tempTensor = initInput;
            for (const auto& layer : _layers) 
            {
                tempTensor = layer->forward(tempTensor); 
            }

            return tempTensor;
        }

        void backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override
        {
            auto tempGrad = outputGradient;
            for (const auto& layer : _layers | std::views::reverse) //some cpp20 shenaningans
            {
                tempGrad = layer->backward(tempGrad, learningRate);
            }
        }

        void save(std::string fileName) override
        {
            std::ofstream outFile;
            outFile.open(fileName, std::ios::binary);
            for (const auto& layer : _layers) 
            {
                layer->save(outFile);
            }
            outFile.close();
        }

        void load(std::string fileName) override
        {
            std::ifstream inFile;
            inFile.open(fileName, std::ios::binary);
            for (const auto& layer : _layers)
            {
                layer->load(inFile);
            }
            inFile.close();
        }
    };
}