#pragma once

#include "core/Layer.hpp"
#include <algorithm>
#include <fstream>

namespace NN::Layers 
{
    template<typename T>
    class ActivationLayer : public NN::Core::Layer<T>
    {
        protected:
            Tensor::Tensor<T> _cacheInput;
            Tensor::Tensor<T> _cachedOutput;     
            Tensor::Tensor<T> _cachedInputGrad;  

        public:
            ActivationLayer() : 
                _cacheInput({1}),
                _cachedOutput({1}),
                _cachedInputGrad({1}) {}; //placeholder

            virtual T activationFunc(T x) = 0;
            virtual T derivativeMath(T x) = 0;

            Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) override
            {
                _cacheInput = input;
                if (_cachedOutput.shape() != input.shape()) [[unlikely]]
                {
                    _cachedOutput = Tensor::Tensor<T>(input.shape());
                }

                std::transform(input.data(), 
                            input.data() + input.size(),
                            _cachedOutput.data(),
                            [this](T val){return this->activationFunc(val);
                            });

                return _cachedOutput;
            }

            Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override
            {
                if (_cachedInputGrad.shape() != outputGradient.shape()) [[unlikely]]
                {
                    _cachedInputGrad = Tensor::Tensor<T>(outputGradient.shape());
                }

                std::transform(outputGradient.data(),
                            outputGradient.data() + outputGradient.size(),
                            _cacheInput.data(),
                            _cachedInputGrad.data(),
                            [this](T gradOut, T cachedIn){return gradOut * this->derivativeMath(cachedIn);
                            });

                return _cachedInputGrad;
            }

            void save(std::ofstream& outFile) override {};
            void load(std::ifstream& infile) override {};
    };
}