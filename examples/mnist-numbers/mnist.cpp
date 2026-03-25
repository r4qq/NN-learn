#include "mnistLoader.hpp"

#define TRAIN_DATA "examples/mnist-numbers/data/train-images.idx3-ubyte"
#define TRAIN_LABELS "examples/mnist-numbers/data/train-labels.idx1-ubyte"

int main()
{
    mnistLoader<void>::loadData(TRAIN_DATA);
    mnistLoader<void>::loadData(TRAIN_LABELS);
    return 0;
}