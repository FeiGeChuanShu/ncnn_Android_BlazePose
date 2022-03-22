#ifndef RELATIVE_VELOCITY_FILTER_H
#define RELATIVE_VELOCITY_FILTER_H

#include "low_pass_filter.h"
#include "time_stamp.h"
#include <deque>
#include <cstdint>
#include <chrono>


class RelativeVelocityFilter {
public:
    enum class DistanceEstimationMode {
        // When the value scale changes, uses a heuristic
        // that is not translation invariant (see the implementation for details).
        kLegacyTransition,
        // The current (i.e. last) value scale is always used for scale estimation.
        // When using this mode, the filter is translation invariant, i.e.
        //     Filter(Data + Offset) = Filter(Data) + Offset.
        kForceCurrentScale,
        
        kDefault = kLegacyTransition
    };
    
public:
    RelativeVelocityFilter(size_t window_size, float velocity_scale, int target_fps,
                           DistanceEstimationMode distance_mode)
    : max_window_size_{window_size},
    window_{window_size},
    velocity_scale_{velocity_scale},
    target_fps_{target_fps},
    distance_mode_{distance_mode} {}
    
    RelativeVelocityFilter(size_t window_size, float velocity_scale, int target_fps)
    : RelativeVelocityFilter{window_size, velocity_scale, target_fps,
        DistanceEstimationMode::kDefault} {}
    
    float Apply(const TimeStamp& timestamp, float value_scale, float value);
    
private:
    struct WindowElement {
        float distance;
        int64_t duration;
    };
    
    float last_value_ = 0.0;
    float last_value_scale_ = 1.0;
    TimeStamp last_timestamp_;
    
    size_t max_window_size_;
    int target_fps_ = 30;
    std::deque<WindowElement> window_;
    LowPassFilter low_pass_filter_{1.0f};
    float velocity_scale_;
    DistanceEstimationMode distance_mode_;
};

#endif // RELATIVE_VELOCITY_FILTER_H
