#include "mnistLoader.hpp"
#include "tensor-hpp/Tensor-simd-block-multi.hpp"
#include "utils/ModelSerialization.hpp"
#include <cmath>
#include <cstdint>

#define TRAIN_DATA "examples/mnist-numbers/data/train-images.idx3-ubyte"
#define TRAIN_LABELS "examples/mnist-numbers/data/train-labels.idx1-ubyte"
#define TRAIN_SPLIT 0.5
#define NUM_TO_DRAW 10 

int main() 
{
    uint64_t batchSize = 256;
    MnistLoader::MnistData<float> mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    uint64_t inputSize = mnistData.learnData.shape()[1];
    uint64_t validImagesAmount = mnistData.testData.shape()[0];
    uint64_t numClasses = mnistData.learnLabels.shape()[1];
    
    auto nn = NN::Utils::Serialization::LoadModel<float>("model.bin");
    
    Tensor::Tensor<float> batchX({batchSize, 784});
    Tensor::Tensor<float> batchY({batchSize, 10});

    std::cout << "Validation dataset size: " << validImagesAmount << " images\n"; 
    auto startTime = std::chrono::high_resolution_clock::now();
    uint64_t correctGuessses = 0;
    for(uint64_t i = 0; i < validImagesAmount; i += batchSize)
    {
        uint64_t currentBatchSize = std::min(batchSize, validImagesAmount - i);
        
        if (batchX.shape()[0] != currentBatchSize) [[unlikely]] 
        {
            batchX = Tensor::Tensor<float>({currentBatchSize, inputSize});
            batchY = Tensor::Tensor<float>({currentBatchSize, numClasses});
        }

         // imgs
        auto xStart = mnistData.testData.data() + (i * inputSize);
        auto xEnd = xStart + (currentBatchSize * inputSize);
        std::copy(xStart, xEnd, batchX.data());
        
        // labels
        auto yStart = mnistData.testLabels.data() + (i * numClasses);
        auto yEnd = yStart + (currentBatchSize * numClasses);
        std::copy(yStart, yEnd, batchY.data());

        auto preds = nn->forward(batchX);

        for (uint64_t j = 0; j < currentBatchSize; ++j) 
        {
            auto predRowStart = preds.data() + (j * numClasses);
            auto trueLabelRow = batchY.data() +  (j * numClasses);

            auto predDigit = std::distance(predRowStart, std::max_element(predRowStart, predRowStart + numClasses));
            auto trueDigit = std::distance(trueLabelRow, std::max_element(trueLabelRow, trueLabelRow + numClasses));

            if (predDigit == trueDigit) 
            {
                correctGuessses += 1;
            }
        }
    }   
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\"Validation\" time: " << duration.count()/1000.0f << "s.\n";
    std::cout << "Validation error: " << 1 - static_cast<float>(correctGuessses) / 
                                         static_cast<float>(validImagesAmount)
                                      << "\n";

    //printing results
    const std::string asciiGreyRamp = " .:#";

    Tensor::Tensor<float> visualX({NUM_TO_DRAW, inputSize});
    Tensor::Tensor<float> visualY({NUM_TO_DRAW, numClasses});

    std::copy(mnistData.testData.data(), mnistData.testData.data() + (NUM_TO_DRAW * inputSize), visualX.data());
    std::copy(mnistData.testLabels.data(), mnistData.testLabels.data() + (NUM_TO_DRAW * numClasses), visualY.data());

    auto visualPreds = nn->forward(visualX);

    for (uint64_t i = 0; i < NUM_TO_DRAW; ++i) 
    {
        auto predRowStart = visualPreds.data() + (i * numClasses);
        auto trueLabelRow = visualY.data() +  (i * numClasses);

        auto predDigit = std::distance(predRowStart, std::max_element(predRowStart, predRowStart + numClasses));
        auto trueDigit = std::distance(trueLabelRow, std::max_element(trueLabelRow, trueLabelRow + numClasses));

        for(uint64_t y = 0; y < 28; ++ y)
        {
            for(uint64_t x = 0; x < 28; ++x)
            {
                uint64_t pixelIdx = (i * 784) + (y * 28) + x;
                float pixelVal = visualX.data()[pixelIdx];
                char asciiIdx = std::round(pixelVal * (asciiGreyRamp.size() - 1));
                std::cout << asciiGreyRamp[asciiIdx];
                std::cout << asciiGreyRamp[asciiIdx];
            }    
            std::cout << "\n";
        }

        std::cout << "Predicted: " << predDigit << ", actual: " << trueDigit << "\n";
    }
    return 0;
}