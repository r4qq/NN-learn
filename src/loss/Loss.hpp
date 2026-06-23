#pragma once

#include "core/config.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>

namespace NN::Loss 
{
    //Mean Squared Error
    template<typename T>
    class MSE
    {
        public:
            static T calculate(const Tensor::Tensor<T>& predicted, const Tensor::Tensor<T>& trueVal)
            {
                T sum = std::transform_reduce(predicted.data(),
                                            predicted.data() + predicted.size(),
                                            trueVal.data(),
                                            T{0},
                                            std::plus<T>{},
                                            [](T predVal, T truVal){
                                                return (predVal - truVal) * (predVal - truVal);
                                            });             
                
                return sum / static_cast<T>(predicted.size());
            } 
            

            static Tensor::Tensor<T> derivative(const Tensor::Tensor<T>& predicted, const Tensor::Tensor<T>& trueVal)
            {
                auto n = predicted.shape()[0];
                Tensor::Tensor<T> gradient(predicted.shape());

                std::transform(predicted.data(),
                            predicted.data() + predicted.size(),
                            trueVal.data(),
                            gradient.data(),
                            [n](T predVal, T truVal){
                                    return T{2} * (predVal - truVal) / static_cast<T>(n);
                            });

                return gradient;
            }
    };

    //Categorical Cross-Entropy Loss
    template<typename T>
    class CCEL
    {
        public:
            static T calculate(const Tensor::Tensor<T>& predicted, const Tensor::Tensor<T>& trueVal) 
            {
                auto batchSize = predicted.shape()[0];
                T sum = std::transform_reduce(trueVal.data(),
                                              trueVal.data() + trueVal.size(),
                                              predicted.data(),
                                              T{0},
                                              std::plus<T>(),
                                              [](T trueVal, T predVal){
                                                return trueVal * std::log(predVal + static_cast<T>(1e-7f));
                                              });
                return -(sum / static_cast<T>(batchSize));
            }

            static Tensor::Tensor<T> derivative(const Tensor::Tensor<T>& predicted, const Tensor::Tensor<T>& trueVal)
            {
                auto batchSize = predicted.shape()[0];
                Tensor::Tensor<T> gradient(predicted.shape());

                std::transform(predicted.data(),
                            predicted.data() + predicted.size(),
                            trueVal.data(),
                            gradient.data(),
                            [batchSize](T predVal, T truVal){
                                    return (predVal - truVal) / static_cast<T>(batchSize);
                            });

                return gradient;
            }
    };
};