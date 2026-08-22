#include <sys/resource.h>
#include <iostream>

namespace NN::Utils::Logging 
{
    inline void PrintMemoryUsage()
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "Peak mem usage: "
                  << usage.ru_maxrss / 1024.0 
                  << " MB\n";
    }
}