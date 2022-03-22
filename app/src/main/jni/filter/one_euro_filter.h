#ifndef ONE_EURO_FILTER_H
#define ONE_EURO_FILTER_H

#include <memory>
#include "time_stamp.h"
#include "low_pass_filter.h"


class OneEuroFilter {
 public:
	
	 OneEuroFilter(double frequency, double min_cutoff, double beta,
		 double derivate_cutoff);
	
	double Apply(const TimeStamp& timestamp, double value_scale, double value);

	private:
	double GetAlpha(double cutoff);

	double frequency_;
	double min_cutoff_;
	double beta_;
	double derivate_cutoff_;
	std::shared_ptr<LowPassFilter> x_;
	std::shared_ptr < LowPassFilter> dx_;
	TimeStamp last_time_;
};

#endif  // ONE_EURO_FILTER_H
