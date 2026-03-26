#include "../src/DenseLayer.hpp"
#include "../src/NeuralNetwork.hpp"
#include "../src/TanhLayer.hpp"
#include "../src/SigmoidLayer.hpp"
#include "../src/Loss.hpp"

#include <cstdint>
#include <memory>
#include <iostream>

#define EPOCHS 10000
#define LEARNINGRATE 0.5f

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
        auto err = Loss::MSE<float>::calculate(res, Y);
        if (i % 1000 == 0) { std::cout <<"In Epoch: " << i << ", Error: " << err << std::endl;}
        auto gradTensor = Loss::
        MSE<float>::derivative(res, Y);
        nn.backward(gradTensor, LEARNINGRATE);
    }
    
    auto predictions = nn.forward(X);
    std::cout << "\nResults:" 
              << "\nIn: 0,0 -> " << predictions(0,0) 
              << "\nIn: 0,1 -> " << predictions(1,0) 
              << "\nIn: 1,0 -> " << predictions(2,0)
              << "\nIn: 1,1 -> " << predictions(3,0) << std::endl;
    return 0;
}