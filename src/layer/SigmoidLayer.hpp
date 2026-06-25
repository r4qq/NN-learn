#pragma once

#include "ActivationLayer.hpp"
#include "LayerType.hpp"
#include <cmath>

namespace NN::Layers 
{
    template<typename T>
    class SigmoidLayer : public NN::Layers::ActivationLayer<T>
    {
        public:
            T activationFunc(T x) override
            {
                return 1 / (1 + std::exp(-x));
            }

            T derivativeMath(T x) override
            {
                T xF = 1 / (1 + std::exp(-x));
                return xF * (1 - xF);
            }

            void save(std::ofstream& outFile) override 
            {
                LayerType type = LayerType::Sigmoid;
                outFile.write(reinterpret_cast<char*>(&type), sizeof(uint32_t));
            };
            void load(std::ifstream& infile) override {};
    };
}