#pragma once

#include "ActivationLayer.hpp"

template<typename T>
class ReLuLayer : public ActivationLayer<T>
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