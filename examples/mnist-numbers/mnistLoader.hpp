#include "../../include/tensor-hpp/Tensor-simd-block.hpp"

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
    Tensor::Tensor<T> loadData(std::string dataPath, std::string labelPath, float trainSplit)
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
        uint8_t tempNum = 0;
        uint32_t magicLabelNumber = 0, labelNum = 0;
        uint8_t label = 0;
        
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
        
        if (dataNum != labelNum) 
        {
            throw std::runtime_error("data and label size mismatch");
        }

        auto trainSize = dataNum * trainSplit;
        
        Tensor::Tensor<T> dataTempTrain({trainSize, rows * cols}); 
        Tensor::Tensor<T> labelTempTrain({trainSize, 10}); 
        labelTempTrain.fill(T{0});

        const T* dataPtr = dataTempTrain.data();
        const T* labelPtr = labelTempTrain.data();
        for (uint64_t i = 0; i < trainSize; ++i) 
        {
            for (uint64_t j = 0; j < cols * rows; ++j) 
            {
                T* iter = dataPtr + ((i * j) + j);
            }
        }

        
        dataFile.close();
        // std::cout << "label: " << static_cast<T>(label) << '\n';
    }
};