#include "layer/DenseLayer.hpp"
#include "layer/LeakyReLuLayer.hpp"
#include "layer/SigmoidLayer.hpp"
#include "loss/Loss.hpp"
#include "models/MLP.hpp"
#include "../mnist-numbers/mnistLoader.hpp"
#include "core/config.hpp"
#include "utils/Logging.hpp"
#include "utils/ModelSerialization.hpp"

#include <algorithm>
#include <cstdint>
#include <chrono>

constexpr auto TRAIN_DATA = "examples/mnist-numbers/data/train-images.idx3-ubyte";
constexpr auto TRAIN_LABELS = "examples/mnist-numbers/data/train-labels.idx1-ubyte";
constexpr auto TRAIN_SPLIT = 0.8;
constexpr auto EPOCHS = 20;
constexpr auto LEARNINGRATE = 0.075;
constexpr uint64_t BATCHSIZE = 256u;

int main()
{

    auto mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    uint64_t trainImagesAmount = mnistData.learnData.shape()[0];
    uint64_t inputSize = mnistData.learnData.shape()[1];

    NN::Models::MLP<float> nn;

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(784, 256));
    nn.addLayer(std::make_unique<NN::Layers::LeakyReLuLayer<float>>());

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(256, 64));
    nn.addLayer(std::make_unique<NN::Layers::LeakyReLuLayer<float>>());

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(64, 16));
    nn.addLayer(std::make_unique<NN::Layers::LeakyReLuLayer<float>>());

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(16, 64));
    nn.addLayer(std::make_unique<NN::Layers::LeakyReLuLayer<float>>());
    
    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(64, 256));
    nn.addLayer(std::make_unique<NN::Layers::LeakyReLuLayer<float>>());

    nn.addLayer(std::make_unique<NN::Layers::DenseLayer<float>>(256, 784));
    nn.addLayer(std::make_unique<NN::Layers::SigmoidLayer<float>>());

    Tensor::Tensor<float> batchX({BATCHSIZE, inputSize});

    std::cout << "Learning loop\n";
    std::cout << "Running " << EPOCHS << " epochs" << " \n";
    std::cout << "Training dataset size: " << trainImagesAmount << " images\n"; 
    std::cout << "Batch size: " << BATCHSIZE << '\n'; 

    auto startTime = std::chrono::high_resolution_clock::now();

    for (uint64_t epoch = 0; epoch < EPOCHS; ++epoch) 
    {
        float epochLoss = 0.0f;
        for (uint64_t j = 0; j < trainImagesAmount; j += BATCHSIZE) 
        {
            uint64_t currentBatchSize = std::min(BATCHSIZE, trainImagesAmount - j);
            
            if (batchX.shape()[0] != currentBatchSize) [[unlikely]] 
            {
                batchX = Tensor::Tensor<float>({currentBatchSize, inputSize});
            }

            auto xStart = mnistData.learnData.data() + (j * inputSize);
            auto xEnd = xStart + (currentBatchSize * inputSize);
            std::copy(xStart, xEnd, batchX.data());

            auto preds = nn.forward(batchX);
            auto loss = NN::Loss::MSE<float>::calculate(preds, batchX);
            auto lossGrad = NN::Loss::MSE<float>::derivative(preds, batchX);
            nn.backward(lossGrad, LEARNINGRATE);

            epochLoss += loss;
        }
        std::cout << "Epoch " << epoch + 1 << " Average Loss: " 
              << epochLoss / (static_cast<float>(trainImagesAmount) / BATCHSIZE) << "\n";
        NN::Utils::Logging::PrintMemoryUsage();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "\"Training\" time: " << duration.count()/1000.0f << "s.\n"; 
    
    uint64_t validImagesAmount = mnistData.testData.shape()[0];
    std::cout << "Validation loop\n";
    std::cout << "Validation dataset size: " << validImagesAmount << " images\n"; 
    std::cout << "Batch size: " << BATCHSIZE << '\n';
    
    float epochLoss = 0.0f;
    startTime = std::chrono::high_resolution_clock::now();

  
    for (uint64_t i = 0; i < validImagesAmount; i += BATCHSIZE) 
    {
        uint64_t currentBatchSize = std::min(BATCHSIZE, validImagesAmount - i);
        
        if (batchX.shape()[0] != currentBatchSize) [[unlikely]] 
        {
            batchX = Tensor::Tensor<float>({currentBatchSize, inputSize});
        }

        auto xStart = mnistData.testData.data() + (i * inputSize);
        auto xEnd = xStart + (currentBatchSize * inputSize);
        std::copy(xStart, xEnd, batchX.data());

        auto preds = nn.forward(batchX);
        auto loss = NN::Loss::MSE<float>::calculate(preds, batchX);
        
        epochLoss += loss;
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "\"Validation\" time: " << duration.count()/1000.0f << "s.\n";
    std::cout << "Validation Average Loss: " 
          << epochLoss / (static_cast<float>(validImagesAmount) / BATCHSIZE) << "\n";

    NN::Utils::Serialization::SaveModel(nn, "model_autoencoder.bin");
}