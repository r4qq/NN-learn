#pragma once

#include "ActivationLayer.hpp"
#include <cmath>

template<typename T>
class SigmoidLayer : public ActivationLayer<T>
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
};