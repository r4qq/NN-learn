#pragma once

#include "ActivationLayer.hpp"
#include <cmath>

template<typename T>
class TanhLayer : public ActivationLayer<T>
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
};