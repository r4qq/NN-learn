#include "Layer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>

template<typename T>
class SoftmaxLayer : public Layer<T>
{
    private:
        Tensor::Tensor<T> _cachedOutput;

    public:
        SoftmaxLayer() : _cachedOutput({1}) {}

        Tensor::Tensor<T> forward(const Tensor::Tensor<T>& input) override
        {
            //Tensor::Tensor<T> outputTensor(input.shape());

            uint64_t batchSize = input.shape()[0];
            uint64_t numClasses = input.shape()[1];

            for (uint64_t i = 0; i < batchSize; ++i) 
            {
                auto M = std::max_element(input.data(), input.data() + (i * numClasses));
                T localSum = std::transform_reduce(input.data(), 
                                                input.data() + input.size(), 
                                                T{0}, 
                                                std::plus<T>(), 
                                                [M](T val){
                                                    std::exp(val - M);
                                                });                
            }

            // for (auto val : input.data()) {
            //     val
            // }
            return _cachedOutput;
        }
};