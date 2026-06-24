#pragma once

#include "core/NeuralNetwork.hpp"
#include <filesystem>
#include <iostream>
#include <sys/resource.h>
#include <string>

namespace NN::Utils 
{
    inline void printMemoryUsage()
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "Peak mem usage: " << usage.ru_maxrss / 1024.0 << " MB\n";
    }

    template<typename T> 
    void saveModel(const NN::Core::NeuralNetwork<T>& nn, std::string fileName) 
    {
        nn.save(fileName);
        std::filesystem::path path = fileName;
        std::cout << "Model saved as: " << fileName 
                                        << " file size: " 
                                        << std::filesystem::file_size(path) / 1000.0f 
                                        << "kB\n";
    }

}