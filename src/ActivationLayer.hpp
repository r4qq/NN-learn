#pragma once

#include "Layer.hpp"
#include <algorithm>

template<typename T>
class ActivationLayer : public Layer<T>
{
    protected:
        Tensor::Tensor<T> _cacheInput;

    public:

        virtual T activationFunc(T x) = 0;
        virtual T derivativeMath(T x) = 0;

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) override
        {
            _cacheInput = input;
            Tensor::Tensor<T> outputTensor(input.shape());

            std::transform(input.data(), 
                           input.data() + input.size(),
                           outputTensor.data(),
                           [this](T val){return this->activationFunc(val);}
                        );

            return outputTensor;
        }

        Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override
        {
            Tensor::Tensor<T> inputGradient(outputGradient.shape());

            std::transform(outputGradient.data(),
                           outputGradient.data() + outputGradient.size(),
                           _cacheInput.data(),
                           inputGradient.data(),
                           [this](T gradOut, T cachedIn){return gradOut * this->derivativeMath(cachedIn);}
                        );

            return inputGradient;
        }
};