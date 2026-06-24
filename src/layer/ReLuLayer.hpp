#pragma once

#include "ActivationLayer.hpp"
#include "LayerType.hpp"

namespace NN::Layers 
{
    template<typename T>
    class ReLuLayer : public NN::Layers::ActivationLayer<T>
    {
        public:
            T activationFunc(T x) override
            {
                return x > 0 ? x : T{0};
            }

            T derivativeMath(T x) override
            {
                return x > 0 ? T{1} : T{0};
            }

            void save(std::ofstream& outFile) override 
            {
                LayerType type = LayerType::ReLu;
                outFile.write(reinterpret_cast<char*>(&type), sizeof(uint32_t));
            };
            void load(std::ifstream& infile) override {};
    };
}