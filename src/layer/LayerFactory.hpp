#pragma once

#include "core/Layer.hpp"
#include "layer/DenseLayer.hpp"
#include "layer/LayerType.hpp"
#include "layer/LeakyReLuLayer.hpp"
#include "layer/ReLuLayer.hpp"
#include "layer/SigmoidLayer.hpp"
#include "layer/TanhLayer.hpp"
#include "layer/SoftmaxLayer.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace NN::Layers 
{
    class LayerFactory
    {
        public:
            template<typename T>
            static std::unique_ptr<NN::Core::Layer<T>> loadNextLayer(std::ifstream &inFile)
            {
                LayerType readType;
                inFile.read(reinterpret_cast<char*>(&readType), sizeof(uint32_t));
                switch (readType) 
                {
                    case LayerType::Dense:
                    {
                        uint64_t inSize, outSize;
                        inFile.read(reinterpret_cast<char*>(&inSize), sizeof(uint64_t));
                        inFile.read(reinterpret_cast<char*>(&outSize), sizeof(uint64_t));
                        auto layerPtr = std::make_unique<NN::Layers::DenseLayer<T>> (inSize, outSize);
                        layerPtr->load(inFile);
                        return layerPtr;
                    }
                    case LayerType::ReLu:
                    {
                        auto layerPtr = std::make_unique<NN::Layers::ReLuLayer<T>>();
                        return layerPtr;
                    }
                    case LayerType::Sigmoid:
                    {
                        auto layerPtr = std::make_unique<NN::Layers::SigmoidLayer<T>>();
                        return layerPtr;
                    }
                    case LayerType::Softmax:
                    {
                        auto layerPtr = std::make_unique<NN::Layers::SoftmaxLayer<T>>();
                        return layerPtr;
                    }
                    case LayerType::Tanh:
                    {
                        auto layerPtr = std::make_unique<NN::Layers::TanhLayer<T>>();
                        return layerPtr;
                    }
                    case LayerType::CCELSoftmax:
                    {
                        auto layerPtr = std::make_unique<NN::Layers::CCELSoftmaxLayer<T>>();
                        return layerPtr;
                    }
                    case LayerType::LeakyReLu:
                    {
                        auto layerPtr = std::make_unique<NN::Layers::LeakyReLuLayer<T>>();
                        return layerPtr;
                    }
                    default:
                        throw std::runtime_error("Unknown Layer type read");
                }
            }
    };
}