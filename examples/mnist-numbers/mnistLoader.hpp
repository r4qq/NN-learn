#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <istream>
#include <string>

template<typename T>
class mnistLoader
{
    public:
        static void loadData(std::string path)
        {
            std::ifstream dataFile(path, std::iostream::binary);
            if (!dataFile.is_open()) 
            {
                std::cout << "there is no file: " << path << '\n';
            }
            uint32_t magicNumber = 0;
            dataFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
            
            std::cout << "data read: "<< magicNumber << '\n';
            std::cout << "swapped data read: "<< std::byteswap(magicNumber) << '\n';
        }
};