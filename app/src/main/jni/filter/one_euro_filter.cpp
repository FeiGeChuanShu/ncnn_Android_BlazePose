#include <cmath>
#include "one_euro_filter.h"
#include "low_pass_filter.h"
using namespace std::chrono;

static const double kEpsilon = 0.000001;
#define M_PI 3.14159265358979323846

OneEuroFilter::OneEuroFilter(double frequency, double min_cutoff, double beta,
	double derivate_cutoff):frequency_(frequency),min_cutoff_(min_cutoff),beta_(beta),
	derivate_cutoff_(derivate_cutoff)
{
	x_ = std::make_shared<LowPassFilter>(GetAlpha(min_cutoff));
	dx_ = std::make_shared<LowPassFilter>(GetAlpha(derivate_cutoff));
	
}
double OneEuroFilter::Apply(const TimeStamp& timestamp, double value_scale,
                            double value) {
  const auto new_timestamp = timestamp;
  if (last_time_ >= new_timestamp) {
    // Results are unpredictable in this case, so nothing to do but
    // return same value
    fprintf(stderr,"New timestamp is equal or less than the last one.");
    return value;
  }

  // update the sampling frequency based on timestamps
  if (!isEmpty(last_time_) && !isEmpty(new_timestamp)) 
  {
    static constexpr double kNanoSecondsToSecond = 1e-9;
	const int64_t duration = duration_cast<nanoseconds>(new_timestamp - last_time_).count();
    frequency_ = 1.0 / (duration * kNanoSecondsToSecond);
  }
  last_time_ = new_timestamp;

  // estimate the current variation per second
  double dvalue = x_->HasLastRawValue()
                      ? (value - x_->LastRawValue()) * value_scale * frequency_
                      : 0.0;  // FIXME: 0.0 or value?
  double edvalue = dx_->ApplyWithAlpha(dvalue, GetAlpha(derivate_cutoff_));
  // use it to update the cutoff frequency
  double cutoff = min_cutoff_ + beta_ * std::fabs(edvalue);

  // filter the given value
  return x_->ApplyWithAlpha(value, GetAlpha(cutoff));
}

double OneEuroFilter::GetAlpha(double cutoff) 
{
	double te = 1.0 / frequency_;
	double tau = 1.0 / (2 * M_PI * cutoff);
	return 1.0 / (1.0 + tau / te);
}



