#include "../../include/tensor-hpp/Tensor-simd-block.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>


namespace MnistLoader
{
    template<typename T>
    struct MnistData 
    {
        Tensor::Tensor<T> learnData;
        Tensor::Tensor<T> learnLabels;
        Tensor::Tensor<T> checkData;
        Tensor::Tensor<T> checkLabels;
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
        
        if (std::byteswap(magicDataNumber)  != 2051 || 
            std::byteswap(magicLabelNumber) != 2049) 
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
        
        for (uint64_t i = 0; i < std::byteswap(rows); i++) 
        {
            for (uint64_t j = 0; j < std::byteswap(cols); j++) 
            {
                dataFile.read(reinterpret_cast<char*>(&tempNum), 1);
                std::cout << (static_cast<T>(tempNum) / 255) << "/";
                // if (tempNum > 128) 
                // {
                //     std::cout << "##";
                // } 
                // else 
                // {
                //     std::cout << "..";
                // }
            }
            std::cout << '\n';
        }
        
        dataFile.close();
        // std::cout << "label: " << static_cast<T>(label) << '\n';
    }
};