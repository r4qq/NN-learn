#include "core/config.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"
#include <stdexcept>

namespace NN::Utils::Image 
{
    template<typename T>
    void loadImage(Tensor::Tensor<T>& imageTensor, const std::string& fileName, int chan)
    {
        int width, height, numChan;
        float* image = stbi_loadf(fileName.c_str(), &width, &height, &numChan, chan); 
        
        if(nullptr == image)
        {
           stbi_image_free(image);
           throw std::runtime_error("couldn't find the image: " + fileName); 
        }

        if(width != imageTensor.shape()[0] || height != imageTensor.shape()[1]) [[unlikely]]
        {
            stbi_image_free(image);
            throw std::runtime_error("loaded image size mismatch");
        }

        auto imageEnd = image + width * height * chan; 
        std::copy(image, imageEnd, imageTensor.data());
        stbi_image_free(image);
    }
}