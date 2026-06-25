#pragma once

#include "core/NeuralNetwork.hpp"
#include "models/MLP.hpp"
#include "models/ModelType.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
#include <string>

namespace NN::Utils 
{
    inline void PrintMemoryUsage()
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "Peak mem usage: "
                  << usage.ru_maxrss / 1024.0 
                  << " MB\n";
    }

    template<typename T> 
    void SaveModel(const NN::Core::NeuralNetwork<T>& nn, std::string fileName) 
    {
        nn.save(fileName);
        std::filesystem::path path = fileName;
        std::cout << "Model saved as: " 
                  << fileName 
                  << " file size: " 
                  << std::filesystem::file_size(path) / 1000.0f 
                  << "kB\n";
    }

    template<typename T>
    std::unique_ptr<NN::Core::NeuralNetwork<T>> LoadModel(std::string fileName)
    {
        std:: ifstream inFile;
        inFile.open(fileName, std::ios::binary);
        if (!inFile.is_open()) 
        {
            throw std::runtime_error("file not found. check path");
        }

        uint32_t readType;
        inFile.read(reinterpret_cast<char*>(&readType), sizeof(uint32_t));
        ModelType type = static_cast<ModelType>(readType);
        
        std::unique_ptr<NN::Core::NeuralNetwork<T>> nn;

        switch (type) 
        {
            case ModelType::MLP :
                nn = std::make_unique<NN::Models::MLP<T>>();
                nn->load(inFile);
                break;
        }
        std::cout << "Model: " << fileName << " loaded\n"; 
        return nn;
    }
}