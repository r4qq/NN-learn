#include <iostream>
#include <sys/resource.h>

namespace Utils 
{
    inline void printMemoryUsage()
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "Peak mem usage: " << usage.ru_maxrss / 1024.0 << " MB\n";
    }
}