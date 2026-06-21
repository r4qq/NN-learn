#include "DenseLayer.hpp"
#include "NeuralNetwork.hpp"
#include "ReLuLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "mnistLoader.hpp"
#include "Loss.hpp"
#include "Utils.hpp"

#include <chrono>
#include <cstdint>

#define TRAIN_DATA "examples/mnist-numbers/data/train-images.idx3-ubyte"
#define TRAIN_LABELS "examples/mnist-numbers/data/train-labels.idx1-ubyte"
#define TRAIN_SPLIT 0.8
#define EPOCHS 10
#define LEARNINGRATE 0.05


int main()
{
    uint64_t batchSize = 64;

    auto mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    
    NeuralNetwork::MLP<float> nn;

    uint64_t totalImages = mnistData.learnData.shape()[0];
    uint64_t inputSize = mnistData.learnData.shape()[1];
    uint64_t numClasses = mnistData.learnLabels.shape()[1];

    nn.addLayer(std::make_unique<DenseLayer<float>>(784, 128));
    nn.addLayer(std::make_unique<ReLuLayer<float>>());

    nn.addLayer(std::make_unique<DenseLayer<float>>(128, 10));
    nn.addLayer(std::make_unique<CCELSoftmaxLayer<float>>());
        
    Tensor::Tensor<float> batchX({batchSize, 784});
    Tensor::Tensor<float> batchY({batchSize, 10});

    std::cout << "Data Shape: " << mnistData.learnData.shape()[0] << "x"
              << mnistData.learnData.shape()[1] << "\n";
    std::cout << "Label Shape: " << mnistData.learnLabels.shape()[0] << "x" 
              << mnistData.learnLabels.shape()[1] << "\n";
    std::cout << "Running " << EPOCHS << " epochs" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (uint64_t epoch = 0; epoch < EPOCHS ; ++epoch) 
    {
        float epochLoss = 0.0;
        for (uint64_t i = 0; i < totalImages; i += batchSize) 
        {
            uint64_t currentBatchSize = std::min(batchSize, totalImages - i);

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
            auto loss = Loss::CCEL<float>::calculate(preds, batchY);
            auto lossGrad = Loss::CCEL<float>::derivative(preds, batchY);
            nn.backward(lossGrad, LEARNINGRATE);

            epochLoss += loss;
        }
        std::cout << "Epoch " << epoch + 1 << " Average Loss: " 
              << epochLoss / (static_cast<float>(totalImages) / batchSize) << "\n";
        Utils::printMemoryUsage();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Learntime: " << duration.count()/1000.0f << "s.\n"; 

    return 0;
}