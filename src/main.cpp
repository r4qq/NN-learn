#include "DenseLayer.hpp"
#include "NeuralNetwork.hpp"
#include "TanhLayer.hpp"
#include "SigmoidLayer.hpp"
#include "Loss.hpp"

#include <cstdint>
#include <memory>
#include <iostream>

#define EPOCHS 100000
#define LEARNINGRATE 0.1f

int main()
{
    //input tensor
    Tensor::Tensor<float> X({4, 2});
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.0;  X(1, 1) = 1.0;
    X(2, 0) = 1.0;  X(2, 1) = 0.0;
    X(3, 0) = 1.0;  X(3, 1) = 1.0;

    // true vals tensor
    Tensor::Tensor<float> Y({4, 1});
    Y(0, 0) = 0.0;
    Y(1, 0) = 1.0;
    Y(2, 0) = 1.0;
    Y(3, 0) = 0.0;

    // the network itself
    NeuralNetwork<float> nn;

    // let's add them layers
    nn.addLayer(std::make_unique<DenseLayer<float>>(2, 3));
    nn.addLayer(std::make_unique<TanhLayer<float>>());
    nn.addLayer(std::make_unique<DenseLayer<float>>(3, 1));
    nn.addLayer(std::make_unique<SigmoidLayer<float>>());

    std::cout << "Running " << EPOCHS << " epochs" << std::endl;
    for (uint64_t i = 0; i < EPOCHS ; ++i) 
    {
        auto res = nn.forward(X);
        auto err = MSE<float>::calculate(res, Y);
        if (i % 1000 == 0) { std::cout << "err: " << err << std::endl;}
        auto gradTensor = MSE<float>::derivative(res, Y);
        nn.backward(gradTensor, LEARNINGRATE);
    }

    return 0;
}