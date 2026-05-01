#include "mnistLoader.hpp"

#define TRAIN_DATA "examples/mnist-numbers/data/train-images.idx3-ubyte"
#define TRAIN_LABELS "examples/mnist-numbers/data/train-labels.idx1-ubyte"
#define TRAIN_SPLIT 0.8

int main()
{
    auto mnistData = MnistLoader::loadData<float>(TRAIN_DATA, TRAIN_LABELS, TRAIN_SPLIT);
    

    return 0;
}