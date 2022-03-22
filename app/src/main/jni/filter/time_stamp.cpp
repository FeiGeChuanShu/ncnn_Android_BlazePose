#include "time_stamp.h"
using namespace std::chrono;

TimeStamp Now() 
{
    return std::chrono::high_resolution_clock::now();
}

bool isEmpty(const TimeStamp& time_stamp) 
{
    return duration_cast<nanoseconds>(time_stamp.time_since_epoch()).count() == 0;
}
