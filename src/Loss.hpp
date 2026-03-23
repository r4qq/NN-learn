#pragma once

#include "../include/tensor-hpp/Tensor.hpp"
#include <algorithm>
#include <numeric>

template<typename T>
class MSE
{
    public:
        static T calculate(const Tensor::Tensor<T>& predicted, const Tensor::Tensor<T>& trueVal)
        {
            auto n = predicted.size();
            T sum = std::transform_reduce(predicted.data(),
                                          predicted.data() + n,
                                          trueVal.data(),
                                          T{0},
                                          std::plus<T>{},
                                          [](T predVal, T truVal){
                                            return (predVal - truVal) * (predVal - truVal);
                                          });             
            
            return sum / n;
        } 
        

        static Tensor::Tensor<T> derivative(const Tensor::Tensor<T>& predicted, const Tensor::Tensor<T>& trueVal)
        {
            auto n = predicted.size();
            Tensor::Tensor<T> gradient(predicted.shape());

            std::transform(predicted.data(),
                           predicted.data() + n,
                           trueVal.data(),
                           gradient.data(),
                           [n](T predVal, T truVal){
                                return T{2} * (predVal - truVal) / n;
                           }    
                        );

            return gradient;
        }
};