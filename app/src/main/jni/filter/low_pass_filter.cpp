#include "low_pass_filter.h"

LowPassFilter::LowPassFilter(float alpha) : initialized_{false} 
{
    SetAlpha(alpha);
}

float LowPassFilter::Apply(float value) 
{
    float result;
    if (initialized_) 
	{
        result = alpha_ * value + (1.0 - alpha_) * stored_value_;
    } 
	else 
	{
        result = value;
        initialized_ = true;
    }
    raw_value_ = value;
    stored_value_ = result;
    return result;
}

float LowPassFilter::ApplyWithAlpha(float value, float alpha) 
{
    SetAlpha(alpha);
    return Apply(value);
}

bool LowPassFilter::HasLastRawValue() 
{ 
	return initialized_; 
}

float LowPassFilter::LastRawValue() 
{ 
	return raw_value_; 
}

float LowPassFilter::LastValue() 
{ 
	return stored_value_; 
}

void LowPassFilter::SetAlpha(float alpha) 
{
    if (alpha < 0.0f || alpha > 1.0f) 
	{
        return;
    }
    alpha_ = alpha;
}
