#pragma once

#include "ActivationLayer.hpp"
#include "LayerType.hpp"
#include <cmath>

namespace NN::Layers 
{
    template<typename T>
    class TanhLayer : public NN::Layers::ActivationLayer<T>
    {
        public:
            T activationFunc(T x) override
            {
                return std::tanh(x);
            }

            T derivativeMath(T x) override
            {
                T xTahn = std::tanh(x);
                return 1 - (xTahn * xTahn);
            }

            void save(std::ofstream& outFile) override 
            {
                LayerType type = LayerType::Tanh;
                outFile.write(reinterpret_cast<char*>(&type), sizeof(uint32_t));
            };
            void load(std::ifstream& infile) override {};
    };
}