# NN-learn

## Note: Project for learning purposes. Not for serious use.

A header-only C++23 machine learning "framework" built from scratch to study core mechanics of neural networks, C++ language, and memory optimization. It relies on the [tensor-hpp](https://github.com/r4qq/tensor-hpp) library for underlying tensor operations and matrix math.

## Key Features

- Layer-Based Architecture: Easily compose networks using a modular addLayer system.
- Components: Includes implementations for DenseLayer, ReLU, Sigmoid, Tanh, and standard Softmax.
- Loss Functions: Mean Squared Error (MSE) and Categorical Cross-Entropy Loss (CCEL). Includes a fused CCELSoftmaxLayer that mathematically cancels the Softmax Jacobian calculation to pass the CCEL gradient directly to the hidden layers.
- Memory optimizations (?): Layers dynamically pre-allocate memory buffers during the first forward/backward pass. Subsequent training iterations overwrite these buffers, avoiding dynamic heap allocations (new/malloc) during the main training loop. Uses C++23 [[unlikely]] attributes to optimize branch prediction for the allocation checks.

## Requirements

- POSIX-compliant OS for <sys/resource.h> memory profiling
- C++ Compiler: C++23
 - tensor-hpp library in the include path (AVX2/FMA and OpenMP support required)

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
