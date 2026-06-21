#include "Layer.hpp"
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

template<typename T>
class SoftmaxLayer : public Layer<T>
{
    private:
        Tensor::Tensor<T> _cachedOutput;
        Tensor::Tensor<T> _cachedInputGrad;

    public:
        SoftmaxLayer() :
            _cachedOutput({1}),
            _cachedInputGrad({1}){} 

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) override
        {
            if (_cachedOutput.shape() != input.shape()) [[unlikely]]
            {
                _cachedOutput = Tensor::Tensor<T>(input.shape()); 
            }

            uint64_t batchSize = input.shape()[0];
            uint64_t numClasses = input.shape()[1];

            for (uint64_t i = 0; i < batchSize; ++i) 
            {
                auto rowStartIn = input.data() + (i * numClasses);
                auto rowEndIn = rowStartIn + numClasses;
            
                auto rowStartOut = _cachedOutput.data() + (i * numClasses);
                auto rowEndOut = rowStartOut + numClasses;
            
                auto maxIter = std::max_element(rowStartIn, rowEndIn);
                T rowMax = *maxIter;

                std::transform(rowStartIn, 
                                rowEndIn, 
                                rowStartOut, 
                                [rowMax](T val){
                                    return std::exp(val - rowMax);
                                });
                
                auto localSum = std::accumulate(rowStartOut, rowEndOut, T{0});

                std::transform(rowStartOut,
                               rowEndOut,
                               rowStartOut,
                               [localSum](T val){
                                   return val / localSum;
                               });                
            }

            return _cachedOutput;
        }

        Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override 
        {
            if (_cachedInputGrad.shape() != outputGradient.shape()) [[unlikely]]
            {
                _cachedInputGrad = Tensor::Tensor<T>(outputGradient.shape()); 
            }

            uint64_t batchSize = outputGradient.shape()[0];
            uint64_t numClasses = outputGradient.shape()[1];

            for (uint64_t i = 0; i < batchSize; ++i) 
            {
                auto sStart = _cachedOutput.data() + (i * numClasses);
                auto sEnd = sStart + numClasses;

                auto gStart = outputGradient.data() + (i * numClasses);
                auto dZStart = _cachedInputGrad.data() + (i * numClasses);

                auto dotRowProduct = std::inner_product(sStart, 
                                                        sEnd, 
                                                        gStart, 
                                                        T{0});

                std::transform(sStart, 
                               sEnd, 
                               gStart, 
                               dZStart,
                               [dotRowProduct](T sVal, T gVal){
                                    return sVal * (gVal - dotRowProduct);
                               });
            }

            return _cachedInputGrad;
        }
};

template<typename T>
class CCELSoftmaxLayer : public Layer<T>
{
    private:
        Tensor::Tensor<T> _cachedOutput;

    public:
        CCELSoftmaxLayer():
            _cachedOutput({1}) {}

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) override
        {
            if (_cachedOutput.shape() != input.shape()) [[unlikely]]
            {
                _cachedOutput = Tensor::Tensor<T>(input.shape()); 
            }

            uint64_t batchSize = input.shape()[0];
            uint64_t numClasses = input.shape()[1];

            for (uint64_t i = 0; i < batchSize; ++i) 
            {
                auto rowStartIn = input.data() + (i * numClasses);
                auto rowEndIn = rowStartIn + numClasses;
            
                auto rowStartOut = _cachedOutput.data() + (i * numClasses);
                auto rowEndOut = rowStartOut + numClasses;
            
                auto maxIter = std::max_element(rowStartIn, rowEndIn);
                T rowMax = *maxIter;

                std::transform(rowStartIn, 
                                rowEndIn, 
                                rowStartOut, 
                                [rowMax](T val){
                                    return std::exp(val - rowMax);
                                });
                
                auto localSum = std::accumulate(rowStartOut, rowEndOut, T{0});

                std::transform(rowStartOut,
                               rowEndOut,
                               rowStartOut,
                               [localSum](T val){
                                   return val / localSum;
                               });                
            }

            return _cachedOutput;
        }

        Tensor::Tensor<T> backward(const Tensor::Tensor<T>& outputGradient, T learningRate) override 
        {
           return outputGradient;
        }
};