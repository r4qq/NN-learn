#include "layer/DenseLayer.hpp"
#include "models/MLP.hpp"
#include "layer/ReLuLayer.hpp"
#include "layer/SoftmaxLayer.hpp"
#include "mnistLoader.hpp"
#include "loss/Loss.hpp"
#include "utils/ModelSerialization.hpp"
#include "utils/Logging.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iterator>

#define TRAIN_DATA "examples/mnist-numbers/data/train-images.idx3-ubyte"
#define TRAIN_LABELS "examples/mnist-numbers/data/train-labels.idx1-ubyte"
#define TRAIN_SPLIT 0.8
#define EPOCHS 10
#define LEARNINGRATE 0.5


int main()
{
    uint64_t batchSize = 256;

    auto mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    
    NN::Models::Sequential<float> nn;

    uint64_t trainImagesAmount = mnistData.learnData.shape()[0];
    uint64_t inputSize = mnistData.learnData.shape()[1];
    uint64_t validImagesAmount = mnistData.testData.shape()[0];
    uint64_t numClasses = mnistData.learnLabels.shape()[1];

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(784, 128));
    nn.addLayer(std::make_unique<NN::Layers::ReLuLayer<float>>());

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(128, 10));
    nn.addLayer(std::make_unique<NN::Layers::CCELSoftmaxLayer<float>>());
        
    Tensor::Tensor<float> batchX({batchSize, 784});
    Tensor::Tensor<float> batchY({batchSize, 10});

    std::cout << "Dataset size: " << trainImagesAmount + validImagesAmount << " images\n";
    std::cout << "Training dataset size: " << trainImagesAmount << " images\n"; 
    std::cout << "Running " << EPOCHS << " epochs" << " \n";
    std::cout << "Batch size: " << batchSize << '\n'; 

    auto startTime = std::chrono::high_resolution_clock::now();

    //train loop
    for (uint64_t epoch = 0; epoch < EPOCHS ; ++epoch) 
    {
        float epochLoss = 0.0f;
        for (uint64_t i = 0; i < trainImagesAmount; i += batchSize) 
        {
            uint64_t currentBatchSize = std::min(batchSize, trainImagesAmount - i);

            if (batchX.shape()[0] != currentBatchSize) [[unlikely]] 
            {
                batchX = Tensor::Tensor<float>({currentBatchSize, inputSize});
                batchY = Tensor::Tensor<float>({currentBatchSize, numClasses});
            }

            // imgs
            auto xStart = mnistData.learnData.data() + (i * inputSize);
            auto xEnd = xStart + (currentBatchSize * inputSize);
            std::copy(xStart, xEnd, batchX.data());

            // labels
            auto yStart = mnistData.learnLabels.data() + (i * numClasses);
            auto yEnd = yStart + (currentBatchSize * numClasses);
            std::copy(yStart, yEnd, batchY.data());

            auto preds = nn.forward(batchX);
            auto loss = NN::Loss::CCEL<float>::calculate(preds, batchY);
            auto lossGrad = NN::Loss::CCEL<float>::derivative(preds, batchY);
            nn.backward(lossGrad, LEARNINGRATE);

            epochLoss += loss;
        }
        std::cout << "Epoch " << epoch + 1 << " Average Loss: " 
              << epochLoss / (static_cast<float>(trainImagesAmount) / batchSize) << "\n";
        NN::Utils::Logging::PrintMemoryUsage();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\"Training\" time: " << duration.count()/1000.0f << "s.\n"; 


    //valid loop
    std::cout << "Validation dataset size: " << validImagesAmount << " images\n"; 
    startTime = std::chrono::high_resolution_clock::now();
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

        auto preds = nn.forward(batchX);

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
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\"Validation\" time: " << duration.count()/1000.0f << "s.\n";
    std::cout << "Validation error: " << 1 - static_cast<float>(correctGuessses) / 
                                         static_cast<float>(validImagesAmount)
                                      << "\n";

    NN::Utils::Serialization::SaveModel(nn, "model_mnist.bin");
    return 0;
}