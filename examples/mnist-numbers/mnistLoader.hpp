#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

template<typename T>
class mnistLoader
{
    public:
        static void loadData(std::string path)
        {
            std::ifstream dataFile;
            dataFile.open(path, std::ios::binary);
            
            if (!dataFile.is_open()) 
            {
                std::cout << "there is no file: " << path << '\n';
                return;
            }
            
            uint32_t magicNumber = 0, num = 0, rows = 0, cols = 0; 
            uint8_t tempNum = 0;
            
            dataFile.read(reinterpret_cast<char*>(&magicNumber), 4);
            
            if (std::byteswap(magicNumber) != 2051) 
            {
                throw std::runtime_error("Invalid mnist data");
            }
            
            dataFile.read(reinterpret_cast<char*>(&num), 4);
            dataFile.read(reinterpret_cast<char*>(&rows), 4);
            dataFile.read(reinterpret_cast<char*>(&cols), 4);

            std::cout << "magic number: "<< std::byteswap(magicNumber) << '\n';
            std::cout << "number of images: "<< std::byteswap(num) << '\n';
            std::cout << "rows: "<< std::byteswap(rows) << '\n';
            std::cout << "cols: "<< std::byteswap(cols) << '\n';
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
        }

        static void loadLabelData(std::string path)
        {
            std::ifstream dataFile;
            dataFile.open(path, std::ios::binary);            
            
            if (!dataFile.is_open()) 
            {
                std::cout << "there is no file: " << path << '\n';
            }

            uint32_t magicNumber = 0, num = 0;
            uint8_t label = 0;
            dataFile.read(reinterpret_cast<char*>(&magicNumber), 4);

            if (std::byteswap(magicNumber) != 2049) 
            {
                throw std::runtime_error("Invalid mnist label data");
            }

            dataFile.read(reinterpret_cast<char*>(&num), 4);
            dataFile.read(reinterpret_cast<char*>(&label), 1);
            
            std::cout << "magic number: "<< std::byteswap(magicNumber) << '\n';
            std::cout << "amount of labels: "<< std::byteswap(num) << '\n';
            std::cout << "label: " << static_cast<T>(label) << '\n';
        }
};