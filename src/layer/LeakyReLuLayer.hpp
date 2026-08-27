#pragma once

#include "ActivationLayer.hpp"
#include "LayerType.hpp"

namespace NN::Layers 
{
    template<typename T>
    class LeakyReLuLayer : public NN::Layers::ActivationLayer<T>
    {
        public:
            T activationFunc(T x) override
            {
                return x > 0 ? x : x * T{0.01};
            }

            T derivativeMath(T x) override
            {
                return x > 0 ? T{1} : T{0.01};
            }

            void save(std::ofstream& outFile) override 
            {
                LayerType type = LayerType::LeakyReLu;
                outFile.write(reinterpret_cast<char*>(&type), sizeof(uint32_t));
            };
            void load(std::ifstream& infile) override {};
    };
}