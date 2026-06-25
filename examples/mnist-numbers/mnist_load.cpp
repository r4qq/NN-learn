#include "mnistLoader.hpp"
#include "utils/Utils.hpp"

#define TRAIN_DATA "examples/mnist-numbers/data/train-images.idx3-ubyte"
#define TRAIN_LABELS "examples/mnist-numbers/data/train-labels.idx1-ubyte"
#define TRAIN_SPLIT 0.5

int main() 
{
    uint64_t batchSize = 256;
    MnistLoader::MnistData<float> mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    uint64_t inputSize = mnistData.learnData.shape()[1];
    uint64_t validImagesAmount = mnistData.testData.shape()[0];
    uint64_t numClasses = mnistData.learnLabels.shape()[1];
    
    auto nn = NN::Utils::LoadModel<float>("model.bin");
    
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

    return 0;
}