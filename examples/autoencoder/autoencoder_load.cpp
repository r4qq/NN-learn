#include "../mnist-numbers/mnistLoader.hpp"
#include "core/config.hpp"
#include "utils/ModelSerialization.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <core/config.hpp>

constexpr auto TRAIN_DATA = "examples/mnist-numbers/data/train-images.idx3-ubyte";
constexpr auto TRAIN_LABELS = "examples/mnist-numbers/data/train-labels.idx1-ubyte";
constexpr auto TRAIN_SPLIT = 0.5;
constexpr auto NUM_TO_DRAW = 10;

int main()
{
    auto mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    uint64_t inputSize = mnistData.learnData.shape()[1];

    Tensor::Tensor<float> noiseData(mnistData.learnData);

    //adding noise
    std:: random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<float> noiseDist(0.0f, 0.2f);

    std::transform(noiseData.data(), noiseData.data() + noiseData.size(), noiseData.data(),
                    [&](float val)
                    {
                        val += noiseDist(gen);
                        val = std::clamp(val, 0.0f, 1.0f);
                        return val;
                    });

    //loading model
    auto nn = NN::Utils::Serialization::LoadModel<float>("model_autoencoder.bin");

    const std::string asciiGreyRamp = ".:-=+*#%@";

    Tensor::Tensor<float> visualNoiseX({NUM_TO_DRAW, inputSize});
    Tensor::Tensor<float> visualX({NUM_TO_DRAW, inputSize});

    std::copy(mnistData.testData.data(), mnistData.testData.data() + (NUM_TO_DRAW * inputSize), visualX.data());

    std::copy(noiseData.data(), noiseData.data() + (NUM_TO_DRAW * inputSize), visualNoiseX.data());

    auto visualPreds = nn->forward(visualNoiseX);

    // le drawings
    for (uint64_t i = 0; i < NUM_TO_DRAW; ++i) 
    {
        std::cout << "NOisy image\n";
        for(uint64_t y = 0; y < 28; ++y)
        {
            for(uint64_t x = 0; x < 28; ++x)
            {
                uint64_t pixelIdx = (i * 784) + (y * 28) + x;
                float pixelVal = visualNoiseX.data()[pixelIdx];
                char asciiIdx = std::round(pixelVal * (asciiGreyRamp.size() - 1));
                std::cout << asciiGreyRamp[asciiIdx] << asciiGreyRamp[asciiIdx];
            }    
            std::cout << "\n";
        }

        std::cout << "Reconstructed image\n";
        for(uint64_t y = 0; y < 28; ++y)
        {
            for(uint64_t x = 0; x < 28; ++x)
            {
                uint64_t pixelIdx = (i * 784) + (y * 28) + x;
                float pixelVal = visualPreds.data()[pixelIdx];
                char asciiIdx = std::round(pixelVal * (asciiGreyRamp.size() - 1));
                std::cout << asciiGreyRamp[asciiIdx] << asciiGreyRamp[asciiIdx];
            } 
            std::cout << "\n";
        }
    }
}