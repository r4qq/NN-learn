#pragma once

#include "core/config.hpp"
#include "core/NeuralNetwork.hpp"
#include "core/Layer.hpp"
#include "ModelType.hpp"
#include "layer/LayerFactory.hpp"
#include <cstdint>
#include <iostream>
#include <memory>
#include <ranges>
#include <utility>


namespace NN::Models 
{
    template<typename T>
    class MLP : public NN::Core::NeuralNetwork<T>
    {
    private:
        std::vector<std::unique_ptr<NN::Core::Layer<T>>> _layers;

    public:
        void addLayer(std::unique_ptr<NN::Core::Layer<T>> layer)
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

        void save(const std::string& fileName) const override 
        {
            std::ofstream outFile;
            outFile.open(fileName, std::ios::binary);
            
            ModelType type = ModelType::MLP;
            outFile.write(reinterpret_cast<char*>(&type), sizeof(uint32_t));

            size_t layersAmount = _layers.size();
            outFile.write(reinterpret_cast<char*>(&layersAmount), sizeof(size_t));

            for (const auto& layer : _layers) 
            {
                layer->save(outFile);
            }
            outFile.close();
        }

        void load(std::ifstream& inFile) override 
        {       
            size_t layersAmount;
            inFile.read(reinterpret_cast<char*>(&layersAmount), sizeof(size_t)); 
        
            for (uint64_t i = 0; i < layersAmount; ++i) 
            {
                auto newLayer = NN::Layers::LayerFactory::loadNextLayer<T>(inFile);
                this->addLayer(std::move(newLayer)); 
            }
            std::cout << "MLP loaded\n";
        }
    };
}