#ifndef TIME_STAMP_H
#define TIME_STAMP_H

#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimeStamp;

// get a timestamp reflecting current moment
TimeStamp Now();
// check if time_stamp is empty(i.e., not a valid time stamp)
bool isEmpty(const TimeStamp& time_stamp);

#endif // TIME_STAMP_H
