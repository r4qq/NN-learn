# NN-learn

## Note: Project for learning purposes. Not for serious use.

A lightweight, header-only C++ machine learning framework built to demonstrate the core mechanics of neural networks, backpropagation, and layers. It is designed to work seamlessly with the tensor-hpp library for high-performance tensor operations.

## Key Features

- Layer-Based Architecture: Easily compose networks using a modular addLayer system.
- Fully Connected (Dense) Layers: Support for standard weights and biases with Normal distribution initialization.
- Activation Functions:
    * ReLU: Rectified Linear Unit for hidden layers.
    * Sigmoid: Classic sigmoid activation for probability-based outputs.
    * Tanh: Hyperbolic tangent activation.
- Automatic Backpropagation: Built-in gradient calculation and weight updates using a simple backward pass.
- Loss Functions: Implementation of Mean Squared Error (MSE) for regression and basic classification tasks.
- Generic Programming: Template-based implementation supports any floating-point type (float, double).
- Modern C++: Leverages C++20 features like std::views::reverse and std::ranges for clean, efficient code.

## Requirements

- Compiler: C++20 or later (required for ranges and views).
- Hardware: AVX2 support is recommended if compiled with tensor-hpp SIMD optimizations.
- Dependencies: Requires tensor-hpp headers in the include path.

## Installation

This is a header-only library. To use it, simply include the src/ directory in your project's include path.

## Usage

### 1. Defining a Network
You can stack layers using std::unique_ptr to build your architecture.

```cpp
#include "NeuralNetwork.hpp"
#include "DenseLayer.hpp"
#include "TanhLayer.hpp"

NeuralNetwork<float> nn;
nn.addLayer(std::make_unique<DenseLayer<float>>(2, 3)); // 2 inputs, 3 neurons
nn.addLayer(std::make_unique<TanhLayer<float>>());
nn.addLayer(std::make_unique<DenseLayer<float>>(3, 1)); // Output layer
```

### 2. Training loop
The framework provides a simple interface for forward and backward passes.

```cpp
for (uint64_t i = 0; i < EPOCHS; ++i) 
{
    // Forward pass
    auto res = nn.forward(X);
    
    // Calculate loss and gradient
    auto err = MSE<float>::calculate(res, Y);
    auto gradTensor = MSE<float>::derivative(res, Y);
    
    // Backpropagate and update weights
    nn.backward(gradTensor, LEARNINGRATE);
}
```

## Building examples
The project uses a Makefile that automatically detects .cpp source files within the examples/ directory and its subfolders.

 * Build all examples: make

 * Build a specific example: make builds/xor (where xor.cpp exists in examples/)

 * Clean: make clean

## License
Project released under the MIT License. See the LICENSE file for details.

---
Author: r4qq (2025-2026)
