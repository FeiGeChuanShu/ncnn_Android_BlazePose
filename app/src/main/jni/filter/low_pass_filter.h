#ifndef LOW_PASS_FILTER_H
#define LOW_PASS_FILTER_H


class LowPassFilter {
public:
    explicit LowPassFilter(float alpha);
    
    float Apply(float value);
    
    float ApplyWithAlpha(float value, float alpha);
    
    bool HasLastRawValue();
    
    float LastRawValue();
    
    float LastValue();
    
private:
    void SetAlpha(float alpha);
    
    float raw_value_;
    float alpha_;
    float stored_value_;
    bool initialized_;
};

#endif // LOW_PASS_FILTER_H
