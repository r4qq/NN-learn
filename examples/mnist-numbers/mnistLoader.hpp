#pragma once

#include "core/config.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#define MNIST_DATA_MAGIC_NUMBER 2051
#define MNIST_LABEL_MAGIC_NUMBER 2049

namespace MnistLoader
{
    template<typename T>
    struct MnistData 
    {
        Tensor::Tensor<T> learnData;
        Tensor::Tensor<T> learnLabels;
        Tensor::Tensor<T> testData;
        Tensor::Tensor<T> testLabels;
    };

    template<typename T>
    MnistData<T> loadData(std::string dataPath, std::string labelPath, float trainSplit)
    {
        std::ifstream dataFile;
        std::ifstream labelFile;
        dataFile.open(dataPath, std::ios::binary);
        labelFile.open(labelPath, std::ios::binary);            
        
        if (!dataFile.is_open() || !labelFile.is_open()) 
        {
            throw std::runtime_error("wrong path, file wans't found. check paths");
        }
        
        uint32_t magicDataNumber = 0, dataNum = 0, rows = 0, cols = 0; 
        uint32_t magicLabelNumber = 0, labelNum = 0;
        
        dataFile.read(reinterpret_cast<char*>(&magicDataNumber), 4);
        labelFile.read(reinterpret_cast<char*>(&magicLabelNumber), 4);
        
        if (std::byteswap(magicDataNumber) != MNIST_DATA_MAGIC_NUMBER || 
            std::byteswap(magicLabelNumber) != MNIST_LABEL_MAGIC_NUMBER) 
        {
            throw std::runtime_error("Invalid mnist data");
        }
        
        dataFile.read(reinterpret_cast<char*>(&dataNum), 4);
        dataFile.read(reinterpret_cast<char*>(&rows), 4);
        dataFile.read(reinterpret_cast<char*>(&cols), 4);
        labelFile.read(reinterpret_cast<char*>(&labelNum), 4);
        
        dataNum = std::byteswap(dataNum);
        rows = std::byteswap(rows);
        cols = std::byteswap(cols);
        labelNum = std::byteswap(labelNum);

        if (dataNum != labelNum) 
        {
            throw std::runtime_error("data and label size mismatch");
        }

        uint64_t trainSize = static_cast<uint64_t>(dataNum * trainSplit);
        uint64_t testSize = dataNum - trainSize;
        uint64_t totalPixels = rows * cols;

        Tensor::Tensor<T> dataTempTrain({trainSize, totalPixels}); 
        Tensor::Tensor<T> labelTempTrain({trainSize, 10}); 
        labelTempTrain.fill(T{0});

        T* trainDataPtr = dataTempTrain.data();
        T* trainLabelPtr = labelTempTrain.data();

        for (uint64_t i = 0; i < trainSize; ++i) 
        {
            uint8_t labelByte = 0;
            labelFile.read(reinterpret_cast<char*>(&labelByte), 1);
            trainLabelPtr[(i * 10) + labelByte] = T{1};

            for (uint64_t j = 0; j < totalPixels; ++j) 
            {
                uint8_t pixelByte = 0;
                dataFile.read(reinterpret_cast<char*>(&pixelByte), 1);
                trainDataPtr[(i * totalPixels) + j] = static_cast<T>(pixelByte) / T{255};
            }
        }

        Tensor::Tensor<T> dataTempTest({testSize, totalPixels}); 
        Tensor::Tensor<T> labelTempTest({testSize, 10}); 
        labelTempTest.fill(T{0});

        T* testDataPtr = dataTempTest.data();
        T* testLabelPtr = labelTempTest.data();

        for (uint64_t i = 0; i < testSize; ++i) 
        {
            uint8_t labelByte = 0;
            labelFile.read(reinterpret_cast<char*>(&labelByte), 1);
            testLabelPtr[(i * 10) + labelByte] = T{1};

            for (uint64_t j = 0; j < totalPixels; ++j) 
            {
                uint8_t pixelByte = 0;
                dataFile.read(reinterpret_cast<char*>(&pixelByte), 1);
                testDataPtr[(i * totalPixels) + j] = static_cast<T>(pixelByte) / T{255};
            }
        }

        dataFile.close();
        labelFile.close();

        return MnistData<T>{
            dataTempTrain,
            labelTempTrain,
            dataTempTest,
            labelTempTest
        };
    }
};