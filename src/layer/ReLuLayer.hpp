#pragma once

#include "ActivationLayer.hpp"

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
    };
}