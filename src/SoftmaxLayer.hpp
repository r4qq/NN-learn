#include "Layer.hpp"
#include <algorithm>
#include <cstdint>

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
                auto M = std::max_element(input.data, input.data() + (i * numClasses));
                
                int total = 0;
                for (uint j = 0; j < 0; ++j) 
                {
                    
                }

            }

            return _cachedOutput;
        }
};